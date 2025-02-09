import logging
from ultralytics import YOLO
import torch
import asyncio
import traceback
import numpy as np
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from modules import Detection, YoloData, Xyxy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger('ultralytics').setLevel(logging.ERROR)


class YoloService:
    """YOLO Service for managing image detection tasks."""
    _instance = None
    _lock = asyncio.Lock()
    _executor = ThreadPoolExecutor(max_workers=8)
    _semaphore = asyncio.Semaphore(8)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YoloService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            print("Initializing YoloService instance")
            self.initialized = False
            self.model = None
            self.device = torch.device('cpu')
            self.Q = asyncio.Queue()
            self.worker_task = None

    async def initialize(self):
        """Initializes the YOLO model and starts the worker task."""
        async with self._lock:
            if not self.initialized:
                print("Loading YOLO model...")
                try:
                    self.model = torch.hub.load(
                        'ultralytics/yolov5', 'yolov5s')
                    # self.model = YOLO("yolov8s.pt")
                    self.model.to(self.device)
                    self.model.eval()
                    self.Q = asyncio.Queue()
                    self.worker_task = asyncio.create_task(
                        self._process_queue())
                    self.initialized = True
                    print("YOLO model loaded successfully")
                except Exception as e:
                    print(f"Error initializing YOLO: {str(e)}")
                    traceback.print_exc()
                    raise

    def _run_model(self, yolo_data: YoloData):
        """Runs the YOLO model with given data."""
        # with torch.no_grad():
        #     results = self.model.predict(
        #         yolo_data.image, conf=yolo_data.confidence, classes=yolo_data.classes, iou=0.6)
        # return results
        self.model.conf = yolo_data.confidence
        self.model.iou = 0.6
        self.model.agnostic_nms = True
        self.model.classes = yolo_data.classes
        return self.model(yolo_data.image)

    async def add_data_to_queue(self, yolo_data: YoloData) -> Optional[List[Detection] | List[List[Detection]]]:
        """Adds an image to the processing queue."""
        if not self.initialized:
            raise RuntimeError("YoloService not initialized")

        try:
            future = asyncio.Future()
            await self.Q.put((yolo_data, future))
            return await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            raise TimeoutError("YOLO processing timeout")
        except Exception as e:
            print(f"Error in add_image_to_queue: {str(e)}")
            traceback.print_exc()
            raise

    async def add_batch_to_queue(self, yolo_data_list: List[YoloData]):
        """
        Adds a batch of images to the processing queue.
        """
        if not self.initialized:
            raise RuntimeError("YoloService not initialized")

        try:
            for yolo_data in yolo_data_list:
                future = asyncio.Future()
                await self.Q.put((yolo_data, future))

            print(f"Added {len(yolo_data_list)} items to the queue.")
        except Exception as e:
            print(f"Error in add_batch_to_queue: {str(e)}")
            traceback.print_exc()
            raise

    async def _process_queue(self):
        """Processes the images in the queue."""
        while True:
            try:
                yolo_data, future = await self.Q.get()
                try:
                    result = await self._process_yolo_data(yolo_data)
                    if not future.done():
                        future.set_result(result)
                except Exception as e:
                    if not future.done():
                        future.set_exception(e)
                finally:
                    self.Q.task_done()
            except Exception as e:
                print(f"Error in queue processor: {str(e)}")
                await asyncio.sleep(1)

    async def _process_yolo_data(self, yolo_data: YoloData) -> List[Dict]:
        """Processes YOLO data for detection."""
        if isinstance(yolo_data.image, list):
            results = await asyncio.gather(
                *[self._run_yolo(img, yolo_data.classes, yolo_data.confidence) for img in yolo_data.image]
            )
            return results
        return await self._run_yolo(yolo_data.image, yolo_data.classes, yolo_data.confidence)

    async def _run_yolo(self, image: np.ndarray, classes: List[int], confidence: float) -> List[Dict]:
        """Runs YOLO on a single image."""
        try:
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(self._executor, self._run_model, YoloData(image=image, confidence=confidence, classes=classes))
            detections = await loop.run_in_executor(self._executor, self._extract_detections, results.xyxy[0], image.shape)
            return detections
        except Exception as e:
            print(f"Error in YOLO detection: {str(e)}")
            traceback.print_exc()
            raise

    def _extract_detections(self, predictions, shape: tuple[int]) -> List[Detection] | List:
        """Applies post-processing on predictions to generate detections."""
        detections = []
        # img_y, img_x = shape[:2]

        for det in predictions:
            try:
                x1, y1, x2, y2, confidence, class_id = det[:6].tolist()

                detections.append(Detection(
                    # bbox=Xyxy(x1=x1/img_x, y1=y1/img_y,
                    #           x2=x2/img_x, y2=y2/img_y),
                    bbox=Xyxy(x1=x1, y1=y1,
                              x2=x2, y2=y2),
                    confidence=f"{confidence:.2f}",
                    class_id=class_id,
                    class_name=self.model.names[class_id])
                )
            except Exception as e:
                print(f"Error processing detection: {str(e)}")

        return detections
