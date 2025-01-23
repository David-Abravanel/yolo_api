import json
import logging
import asyncio
import aioboto3
import numpy as np
from datetime import datetime
from services import YoloService
from typing import List, Dict, Optional
from services.api_service import APIService
from services.mask_service import MaskService
from modules import AlertsRequest, AlertsResponse
from modules.models import DetectionEncoder, YoloData


class SQSService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        region: str,
        data_for_queue_url: str,
        backend_queue_url: str,
        yolo_service: YoloService,
        batch_size: int = 20,
    ):
        if not hasattr(self, 'initialized'):
            self.logger = logging.getLogger(self.__class__.__name__)
            self.session = aioboto3.Session(region_name=region)
            self._sqs_client = None
            self.data_for_queue_url = data_for_queue_url
            self.backend_queue_url = backend_queue_url
            self.yolo = yolo_service
            self.batch_size = batch_size
            self._num_of_receives = 0
            self._num_of_sends = 0
            self._metrics = {
                'receives': 0,
                'sends': 0,
                'no_motion': 0,
                'expires': 0,
                'errors': 0
            }
            self.time_process: Optional[datetime] = None
            self.initialized = True

    async def get_sqs_client(self):
        """
        Returns an active SQS client. Creates a new one if it does not exist or is closed.
        """
        try:
            if not hasattr(self, '_sqs_client') or self._sqs_client is None:
                async with self.session.client('sqs') as sqs:
                    self._sqs_client = sqs
            return self._sqs_client
        except Exception as e:
            self.logger.error(f"Error creating or retrieving SQS client: {e}")
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
            self._metrics['receives'] += len(messages)
            return messages
        except Exception as e:
            self.logger.error(f"SQS receive error: {e}")
            self._metrics['errors'] += 1
            return []

    async def send_message(self, detection_data: Dict) -> bool:
        try:
            sqs = await self.get_sqs_client()
            await sqs.send_message(
                QueueUrl=self.backend_queue_url,
                MessageBody=json.dumps(detection_data, cls=DetectionEncoder)
            )
            self._metrics['sends'] += 1
            return True
        except Exception as e:
            self.logger.error(f"SQS send error: {e}")
            self._metrics['errors'] += 1
            return False

    async def delete_message(self, receipt_handle: str) -> bool:
        try:
            sqs = await self.get_sqs_client()
            await sqs.delete_message(
                QueueUrl=self.data_for_queue_url,
                ReceiptHandle=receipt_handle
            )
            return True
        except Exception as e:
            self.logger.error(f"SQS delete error: {e}")
            self._metrics['errors'] += 1
            return False

    async def process_message(self, message: Dict):
        try:
            message_body: AlertsRequest = AlertsRequest(
                **json.loads(message['Body']))
            S3urls = message_body.snapshots
            camera_data = message_body.camera_data
            frames = [await APIService.fetch_image(url) for url in S3urls]

            # Add authentication or pre-signed URL logic if needed
            if not np.any(frames):
                print("No valid frames retrieved -> type:", 'xml')
                await self.delete_message(message['ReceiptHandle'])
                self._metrics['expires'] += 1
                return

            mask = MaskService.create_combined_mask(
                frames[0].shape, camera_data.masks, camera_data.is_focus)

            # Handle motion detection if needed
            if len(frames) > 1:
                if not MaskService.detect_significant_movement(frames, mask):
                    print('No motion')
                    self._metrics['no_motion'] += 1
                    await self.delete_message(message['ReceiptHandle'])
                    return

            yolo_data = YoloData(
                image=frames,
                confidence=camera_data.confidence,
                classes=camera_data.classes
            )
            detection_result = await self.yolo.add_data_to_queue(yolo_data=yolo_data)
            if detection_result and any(detection_result):
                print('Object detect!!!')
                # Create detection response
                detection = AlertsResponse(
                    camera_data=message_body.without_camera_data(),
                    detections=detection_result
                )
                MaskService.print_results(detection_result)

                # # Send detection result
                await self.send_message(detection)
            else:
                print('No Object Detected')

            # Delete processed message
            await self.delete_message(message['ReceiptHandle'])

            # Add any processing logic here
        except Exception as e:
            self.logger.error(f"Message processing error: {e}")

    async def continuous_transfer(self, poll_interval: int = 0):
        while True:
            try:
                messages = await self.get_messages()
                print(f"Receive: {len(messages)} motions")
                if messages:
                    tasks = [self.process_message(msg) for msg in messages]
                    await asyncio.gather(*tasks, return_exceptions=True)

                await asyncio.sleep(poll_interval)

            except Exception as e:
                self.logger.error(f"Transfer loop error: {e}")
                await asyncio.sleep(poll_interval)

    def get_metrics(self) -> Dict:
        return self._metrics.copy()
