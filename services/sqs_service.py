import json
import logging
import asyncio
import aioboto3
import numpy as np
from datetime import datetime
from typing import List, Dict
from pydantic import ValidationError

from services import YoloService, MaskService, APIService
from modules import AlertsRequest, AlertsResponse, DetectionEncoder, YoloData, metrics_tracker


class SQSService:
    def __init__(
        self,
        region: str,
        data_for_queue_url: str,
        backend_queue_url: str,
        yolo_service: YoloService,
        batch_size: int = 20,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())

        self.session = aioboto3.Session(region_name=region)
        self._sqs_client = None
        self.data_for_queue_url = data_for_queue_url
        self.backend_queue_url = backend_queue_url
        self.yolo = yolo_service
        self.batch_size = batch_size

    async def get_sqs_client(self):
        try:
            if self._sqs_client is None:
                self._sqs_client = await self.session.client('sqs').__aenter__()
            return self._sqs_client
        except Exception as e:
            self.logger.error(
                "Error creating or retrieving SQS client", exc_info=True)
            self._sqs_client = None
            raise

    async def get_messages(self) -> List[Dict]:
        try:
            sqs = await self.get_sqs_client()
            response = await sqs.receive_message(
                QueueUrl=self.data_for_queue_url,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=5,
                VisibilityTimeout=10
            )
            messages = response.get('Messages', [])
            await metrics_tracker.update('receives', len(messages))
            await metrics_tracker.update('message_in_action', len(messages))
            return messages
        except Exception as e:
            self.logger.error(
                "Failed to receive messages from SQS", exc_info=True)
            await metrics_tracker.update('errors', {'get': 1})
            return []

    async def send_message(self, detection_data: AlertsResponse) -> bool:
        try:
            sqs = await self.get_sqs_client()
            await sqs.send_message(
                QueueUrl=self.backend_queue_url,
                MessageBody=json.dumps(detection_data, cls=DetectionEncoder)
            )
            self.logger.info(f"✅ Successfully sent detection data")
            return True
        except Exception as e:
            self.logger.error("❌ Failed to send message to SQS", exc_info=True)
            await metrics_tracker.update('errors', {'send': 1})
            return False

    async def delete_message(self, receipt_handle: str) -> bool:
        try:
            sqs = await self.get_sqs_client()
            await sqs.delete_message(
                QueueUrl=self.data_for_queue_url,
                ReceiptHandle=receipt_handle
            )
            await metrics_tracker.update('message_in_action', -1)
            return True
        except Exception as e:
            self.logger.error(
                "❌ Failed to delete message from SQS", exc_info=True)
            await metrics_tracker.update('errors', {'delete': 1})
            return False

    async def process_message(self, message: Dict):
        start_time = datetime.now()
        try:
            try:
                message_body = AlertsRequest(**json.loads(message['Body']))
            except ValidationError as ve:
                self.logger.warning(f"⚠️ Validation error for message: {
                                    message['MessageId']}", exc_info=True)
                await self.delete_message(message['ReceiptHandle'])
                return

            S3urls = message_body.snapshots
            camera_data = message_body.camera_data
            frames = await asyncio.gather(*[APIService.fetch_image(url) for url in S3urls], return_exceptions=True)

            if not all(isinstance(frame, np.ndarray) for frame in frames):
                self.logger.warning(f"❌ Expired or invalid image URLs for message: {
                                    message['MessageId']}")
                await self.delete_message(message['ReceiptHandle'])
                await metrics_tracker.update('expires')
                detection_happened = False
                await metrics_tracker.add_processing_time(
                    (datetime.now() - start_time).total_seconds(),
                    detection_happened
                )
                return

            mask = MaskService.create_combined_mask(
                frames[0].shape, camera_data.masks, camera_data.is_focus)

            if len(frames) > 1 and isinstance(mask, np.ndarray) and not MaskService.detect_significant_movement(frames, mask):
                self.logger.info(f"🛑 No significant movement detected")
                await self.delete_message(message['ReceiptHandle'])
                await metrics_tracker.update('no_motion')
                detection_happened = False
                await metrics_tracker.add_processing_time(
                    (datetime.now() - start_time).total_seconds(),
                    detection_happened
                )
                return

            yolo_data = YoloData(
                image=frames,
                confidence=camera_data.confidence,
                classes=camera_data.classes
            )
            detection_result = await self.yolo.add_data_to_queue(yolo_data=yolo_data)
            detections = [MaskService.get_detections_on_mask(
                det, mask, frames[0].shape) for det in detection_result]

            if detections and any(detections):
                detection = AlertsResponse(
                    camera_data=message_body.without_camera_data(),
                    detections=detections
                )
                camera_event_time = datetime.fromisoformat(
                    str(detection.camera_data.event_time))
                detection_time = datetime.now(camera_event_time.tzinfo)
                camera_to_detection_time = (
                    detection_time - camera_event_time).total_seconds()

                detection_happened = True
                await metrics_tracker.add_processing_time(
                    (detection_time - start_time.astimezone(camera_event_time.tzinfo)
                     ).total_seconds(),
                    detection_happened
                )
                await metrics_tracker.add_camera_detection_time(camera_to_detection_time)

                MaskService.print_results(detections)
                await self.send_message(detection)
            else:
                if detection_result and any(detection_result):
                    self.logger.info(f"🔍 Detection outside the mask")
                    await metrics_tracker.update('no_detection_on_mask')
                else:
                    self.logger.info(f"🚶 Movement but No detection")
                    await metrics_tracker.update('no_detection')

                detection_happened = False
                await metrics_tracker.add_processing_time(
                    (datetime.now() - start_time).total_seconds(),
                    detection_happened
                )

            await self.delete_message(message['ReceiptHandle'])

        except Exception as e:
            await metrics_tracker.update('errors', {'general': 1})
            await metrics_tracker.update('message_in_action', -1)
            self.logger.error(f"❌ Error processing message: {
                              message.get('MessageId', 'Unknown ID')}", exc_info=True)

    async def continuous_transfer(self, poll_interval: int = 0):
        while True:
            try:
                messages = await self.get_messages()
                if len(messages) > 0:
                    self.logger.info(
                        f"📬 Processing {len(messages)} messages")
                if messages:
                    tasks = [self.process_message(msg) for msg in messages]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    for result in results:
                        if isinstance(result, Exception):
                            self.logger.error(
                                "❌ Error while processing a message", exc_info=result)

                await asyncio.sleep(poll_interval)

            except Exception as e:
                self.logger.error(
                    "❌ Error in continuous transfer loop", exc_info=True)
                await asyncio.sleep(poll_interval)

    async def get_metrics(self) -> Dict:
        return metrics_tracker.calculate_metrics()
