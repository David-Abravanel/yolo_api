# @app.post("/yolo-detect/single", response_model=AlertResponse)
# async def check_single_picture(request: AlertRequest):
#     """
#     Process a single image and detect objects using YOLO.

#     Args:
#     - request: Image URL and optional camera configuration (confidence, classes).

#     Returns:
#     - AlertResponse: Detection results with URL, camera data, and detected objects.
#     """
#     camera_data: CameraData = request.camera_data
#     url: str = request.url

#     try:
#         # Fetch image from URL
#         image: np.ndarray = await S3Service.fetch_image(url)

#         # Prepare YOLO data for detection
#         yolo_data: YoloData = YoloData(
#             image=image, confidence=camera_data.confidence, classes=camera_data.classes)

#         # Process image with YOLO
#         detections = await yolo_service.add_image_to_queue(yolo_data)

#         # Print results
#         MaskService.print_results(detections)

#         return AlertResponse(url=url, camera_data=camera_data, detections=detections)

#     except Exception as e:
#         print(f"Error processing alert: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/yolo-detect/single-with-mask", response_model=AlertResponse)
# async def check_single_picture_with_mask(request: AlertRequest):
#     """
#     Process a single image with a mask and detect objects using YOLO.

#     Args:
#     - request: Image URL and camera configuration (confidence, classes, masks, focus).

#     Returns:
#     - AlertResponse: Detection results with mask filtering.
#     """
#     camera_data: CameraData = request.camera_data
#     url: str = request.url

#     try:
#         # Fetch image from URL
#         image: np.ndarray = await S3Service.fetch_image(url)

#         # Prepare YOLO data for detection
#         yolo_data: YoloData = YoloData(
#             image=image, confidence=camera_data.confidence, classes=camera_data.classes)

#         # Process image with YOLO
#         detections = await yolo_service.add_image_to_queue(yolo_data)

#         # Generate mask and filter detections
#         mask = MaskService.create_combined_mask(
#             image.shape, camera_data.masks, camera_data.is_focus)
#         detections = MaskService.get_detections_on_mask(
#             detections, mask, image.shape)

#         # Print results
#         MaskService.print_results(detections)

#         return AlertResponse(url=url, camera_data=camera_data, detections=detections)

#     except Exception as e:
#         print(f"Error processing alert: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/yolo-detect/group", response_model=AlertsResponse)
# async def check_many_pictures(request: AlertsRequest):
#     """
#     Process multiple images and detect objects using YOLO.

#     Args:
#     - request: List of image URLs and optional camera configuration (confidence, classes).

#     Returns:
#     - AlertsResponse: Detection results for all images.
#     """
#     camera_data: CameraData = request.camera_data
#     urls: List[str] = request.urls

#     try:
#         # Fetch images from URLs
#         frames = [await S3Service.fetch_image(url) for url in urls]

#         # Prepare YOLO data for detection
#         yolo_data = YoloData(
#             image=frames, classes=camera_data.classes, confidence=camera_data.confidence)

#         # Process with YOLO
#         detections = await yolo_service.add_image_to_queue(yolo_data)

#         # Print results
#         MaskService.print_results(detections)

#         return AlertsResponse(urls=urls, camera_data=camera_data, detections=detections)

#     except Exception as e:
#         print(f"Error processing alert: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/yolo-detect/group-with-mask", response_model=Optional[AlertsResponse | dict])
# async def check_many_pictures_with_mask(request: AlertsRequest):
#     """
#     Process multiple images with a mask and detect objects using YOLO.

#     Args:
#     - request: List of image URLs and camera configuration (confidence, classes, masks, focus).

#     Returns:
#     - AlertsResponse: Detection results filtered by mask.
#     - dict: Error message if no masks provided.
#     """
#     if not request.camera_data.masks:
#         return {"message": "Masks must be provided."}

#     camera_data: CameraData = request.camera_data
#     urls: List[str] = request.urls

#     try:
#         # Fetch images from URLs
#         frames = [await S3Service.fetch_image(url) for url in urls]

#         # Prepare YOLO data for detection
#         yolo_data = YoloData(
#             image=frames, classes=camera_data.classes, confidence=camera_data.confidence)

#         # Process with YOLO
#         detections = await yolo_service.add_image_to_queue(yolo_data)

#         # Generate mask and filter detections
#         mask = MaskService.create_combined_mask(
#             frames[0].shape, camera_data.masks, camera_data.is_focus)
#         detections = [MaskService.get_detections_on_mask(
#             det, mask, frames[0].shape) for det in detections]

#         # Print results
#         MaskService.print_results(detections)

#         return AlertsResponse(urls=urls, camera_data=camera_data, detections=detections)

#     except Exception as e:
#         print(f"Error processing alert: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/yolo-detect/group-with-mask-and-motion", response_model=Optional[AlertsResponse | dict])
# async def check_many_pictures_with_mask_and_motion(request: AlertsRequest):
#     """
#     Process multiple images with a mask and motion detection using YOLO.

#     Args:
#     - request: List of image URLs and camera configuration (confidence, classes, masks, focus).

#     Returns:
#     - AlertsResponse: Detection results with motion filtering.
#     - dict: Message if no motion detected.
#     """
#     camera_data: CameraData = request.camera_data
#     urls: List[str] = request.urls

#     try:
#         # Fetch images from URLs
#         frames = [await S3Service.fetch_image(url) for url in urls]

#         # Generate mask
#         mask = MaskService.create_combined_mask(
#             frames[0].shape, camera_data.masks, camera_data.is_focus)

#         # Check for significant motion
#         movement_detected = MaskService.detect_significant_movement(
#             frames, mask)
#         if not movement_detected:
#             return {"message": "No significant motion detected."}

#         # Prepare YOLO data for detection
#         yolo_data = YoloData(
#             image=frames, classes=camera_data.classes, confidence=camera_data.confidence)

#         # Process with YOLO
#         detections = await yolo_service.add_image_to_queue(yolo_data)

#         # Filter detections
#         detections = [MaskService.get_detections_on_mask(
#             det, mask, frames[0].shape) for det in detections]

#         # Print results
#         MaskService.print_results(detections)

#         return AlertsResponse(urls=urls, camera_data=camera_data, detections=detections)

#     except Exception as e:
#         print(f"Error processing alert: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/yolo-detect", response_model=Optional[Union[AResponse, Dict[str, str]]])
# async def generic_detection(request: Optional[ARequest], motion: Optional[bool] = Query("true")):
#     """
#     Detect objects in images using YOLO, with optional motion detection.

#     Args:
#     - request: Request containing image URL(s) and camera configuration, (confidence, classes, masks, focus).
#     - motion: Whether to enable motion detection (default: True).
#     - example: `http://localhost/yolo-detect/?motion=true`

#     Returns:
#     - AlertsResponse: Detection results with URLs, camera settings, and bounding box data.
#     - Dict[str, str]: Message if no significant motion is detected.

#     Raises:
#     - HTTPException: For image decoding or internal errors.
#     """

#     # Get URLs and camera data
#     print('data received')
#     urls = [request.url] if hasattr(request, 'url') else request.urls
#     camera_data = request.camera_data

#     try:
#         # Fetch images
#         frames = [await S3Service.fetch_image(url) for url in urls]
#         if not frames:
#             raise HTTPException(
#                 status_code=400, detail="Failed to decode image")

#         # Create combined mask
#         mask = MaskService.create_combined_mask(
#             frames[0].shape, camera_data.masks, camera_data.is_focus)

#         # Handle motion detection if needed
#         if len(frames) > 1 and bool(motion):
#             T, mask = MaskService.detect_significant_movement(frames, mask)
#             if not T:
#                 return {"message": "No significant movement detected. Use '/yolo-detect?motion=false' for batch detection."}
#         # Prepare YOLO data
#         yolo_data = YoloData(
#             image=frames, classes=camera_data.classes, confidence=camera_data.confidence)
#         detections = await yolo_service.add_data_to_queue(yolo_data=yolo_data)

#         # Adjust detections for mask
#         detections = [MaskService.get_detections_on_mask(
#             det, mask, frames[0].shape) for det in detections]

#         # Return the detection results
#         return AResponse(urls=urls, camera_data=camera_data, detections=detections)

#     except Exception as e:
#         print(f"Error processing alert: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

import os
# from typing import Dict, Optional, Union
import uvicorn
import asyncio
from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
# from fastapi import FastAPI, HTTPException, Query

from services import YoloService, SQSService, S3Service, load_AWS_env

# from modules.models import ARequest, AResponse, AlertsRequest, AlertsResponse, YoloData
from modules import metrics_tracker

load_AWS_env(secret_name='ILG-YOLO-SQS')

yolo_service = YoloService()

queue_for_yolo_url = os.getenv('queue_for_yolo_url')
queue_for_backend_url = os.getenv('queue_for_backend_url')
bucket_name = os.getenv('bucket_name')
# local_coll = os.getenv('local_coll')
images_folder = os.getenv('images_folder')
region = os.getenv('region')

s3Service = S3Service(Bucket=bucket_name, Folder=images_folder, region=region)

SqsService = SQSService(
    region=region,
    data_for_queue_url=queue_for_yolo_url,
    backend_queue_url=queue_for_backend_url,
    S3Service=s3Service,
    yolo_service=yolo_service,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await s3Service.initialize()
    await yolo_service.initialize()
    task = asyncio.create_task(SqsService.continuous_transfer())

    try:
        yield
    finally:
        task.cancel()

app = FastAPI(title="YOLO Detection Service", lifespan=lifespan)


@app.get("/health")
async def get_metric(request: Request):
    metrics = metrics_tracker.calculate_metrics()
    return {"data": metrics, "status": 'healthy'}


@app.get("/health-clients")
def get_client_metric():
    return {"data": metrics_tracker.clients_track, "status": 'healthy'}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        # workers=1,
        reload=True
    )
