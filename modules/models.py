from collections import defaultdict
import json
import asyncio
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pydantic import BaseModel, field_validator, model_validator


class Coordinates(BaseModel):
    x: float
    y: float


class Shape(BaseModel):
    shape: List[Coordinates]
    # shape: List[Coordinates] = [Coordinates(x=0, y=0), Coordinates(
    #     x=0.5, y=0), Coordinates(x=0.5, y=1), Coordinates(x=0, y=1)]


class Xyxy(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @field_validator("x1", "y1", "x2", "y2", mode="before")
    def round_to_three_decimals(cls, value):
        return round(value, 3)

    @model_validator(mode="after")
    def validate_bbox(cls, values):
        if values.x1 >= values.x2 or values.y1 >= values.y2:
            raise ValueError(
                "Invalid bounding box: x1 must be < x2 and y1 must be < y2")
        return values

    def update(self, **kwargs):
        updated_data = self.model_dump()
        updated_data.update(kwargs)
        return self.model_validate(updated_data)


class YoloData(BaseModel):
    image: np.ndarray | List[np.ndarray]
    confidence: float
    classes: List[int]

    class Config:
        arbitrary_types_allowed = True


class CameraData(BaseModel):
    classes: List[int] = [0, 1, 2, 3, 5, 7]
    confidence: float = 0.35
    masks: Optional[List[Shape]] = []
    is_focus: bool = True


class Detection(BaseModel):
    bbox: Xyxy
    confidence: float
    class_id: int
    class_name: str


class Request(BaseModel):
    ip: str
    nvr_name: str
    channel_id: str
    event_time: datetime
    event_type: str
    snapshots: List[str]
    temp_urls: List[str]
    time_detect: datetime


# class ARequest(BaseModel):
#     urls: List[str]
#     camera_data: CameraData = CameraData()


class AlertsRequest(BaseModel):
    ip: str
    nvr_name: str
    channel_id: str
    event_time: datetime
    event_type: str
    snapshots: List[str]
    temp_urls: List[str] = []
    camera_data: CameraData = CameraData()

    def without_camera_data(self):
        # Create a dictionary without `camera_data` and add `time_detect`
        data = self.dict(exclude={"camera_data"})
        data["time_detect"] = str(datetime.now())  # Add current time
        return data


class AlertsResponse(BaseModel):
    camera_data: Request
    detections: List[List[Detection]] | List


# class AResponse(BaseModel):
#     urls: List[str]
#     camera_data: CameraData
#     detections: List[List[Detection]] | List


class DetectionEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


@dataclass
class MetricsTracker:
    start_time: datetime = field(default_factory=datetime.now)
    receives: int = 0
    sends: int = 0
    Alert_in_action: int = 0
    motion_mask_time: List[float] = field(default_factory=list)
    no_motion: int = 0
    no_detection: int = 0
    no_detection_on_mask: int = 0
    expires: int = 0
    logs: List[Dict] = field(default_factory=list)
    errors: dict = field(default_factory=lambda: {
        'get': 0, 'send': 0, 'delete': 0, 'general': 0
    })

    clients_track: Dict[str, Dict[int, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    processing_times: List[float] = field(default_factory=list)
    camera_to_detection_times: List[float] = field(default_factory=list)

    _metrics_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def update_client(self, client_ip: str, client_channel: int):
        async with self._metrics_lock:
            self.clients_track[client_ip][client_channel] += 1

    async def update(self, metric_type: str, value: int = 1, err: Dict = {}):
        async with self._metrics_lock:
            if hasattr(self, metric_type):
                current_value = getattr(self, metric_type)

                if isinstance(current_value, dict):
                    for key, val in value.items():
                        if key in current_value:
                            current_value[key] += val
                        else:
                            current_value[key] = val
                    if "errors" in current_value:
                        self.logs.append(err)
                else:
                    setattr(self, metric_type, current_value + value)

    async def add_detect_motion_time(self, time: float):
        async with self._metrics_lock:
            self.motion_mask_time.append(time)

    async def add_processing_time(self, time: float, detection: bool = False):
        async with self._metrics_lock:
            self.processing_times.append(time)
            if detection:
                self.sends += 1

    async def process_detection_time(self, camera_start_event_time: str, start_time: datetime, detection_happened: bool = False):
        camera_event_time = datetime.fromisoformat(
            str(camera_start_event_time))
        detection_time_camera_zon = datetime.now(camera_event_time.tzinfo)
        camera_to_detection_time = (
            detection_time_camera_zon - camera_event_time).total_seconds()

        await self.add_processing_time(
            (datetime.now() - start_time).total_seconds(),
            detection_happened
        )

        await self.add_camera_detection_time(camera_to_detection_time)

    async def add_camera_detection_time(self, time: float):
        async with self._metrics_lock:
            self.camera_to_detection_times.append(time)

    def calculate_metrics(self) -> dict:
        total_run_time = datetime.now() - self.start_time
        total_send_attempts = self.receives

        def format_time(time_delta: timedelta, ms: bool = False) -> str:
            units = [('Y', 365*24*60*60), ('M', 30*24*60*60), ('D', 24*60*60),
                     ('h', 3600), ('m', 60), ('s', 1)]
            if ms:
                units.append(('ms', 0.001))

            total_seconds = time_delta.total_seconds()
            result = []

            for unit, divisor in units:
                if unit == 'ms' and ms:
                    ms_value = (total_seconds - int(total_seconds)) * 1000
                    if ms_value > 0:
                        result.append(f"{int(ms_value)}{unit}")
                else:
                    unit_qty = int(total_seconds // divisor)
                    if unit_qty > 0 or result:
                        result.append(f"{unit_qty}{unit}")
                    total_seconds -= unit_qty * divisor

            return ' '.join(result) if result else '0s'

        def calculate_time_stats(times: List[float]) -> dict:
            return {
                'avg': format_time(timedelta(seconds=np.mean(times) if times else 0)),
                'median': format_time(timedelta(seconds=np.median(times) if times else 0)),
                'std': format_time(timedelta(seconds=np.std(times) if times else 0))
            }

        def calculate_rate(part: int, total: int) -> float:
            return round((part / total * 100) if total else 0, 3)

        total_errors = sum(self.errors.values())

        total_run_time_str = format_time(total_run_time)
        work_run_time_str = format_time(
            timedelta(seconds=sum(self.processing_times)))

        return {
            '🕒 total run time': total_run_time_str,
            '🔧 work run time': work_run_time_str,
            '📥 receives': self.receives,
            '✅ sends': self.sends,
            '🔄 Alert in action': self.Alert_in_action,
            '🚫🚶 no movement': self.no_motion,
            '❌🔍 no detection': self.no_detection,
            '🚫🎭 no detection on mask': self.no_detection_on_mask,
            '⌛ expires': self.expires,
            '⚠️ errors': self.errors,
            '🚶🎭 detect motion mask time': format_time(timedelta(seconds=np.mean(self.motion_mask_time) if self.motion_mask_time else 0), ms=True),
            '⚖️ avg detection time': format_time(timedelta(seconds=np.mean(self.processing_times) if self.processing_times else 0), ms=True),
            '⏱️ camera to detection times': calculate_time_stats(self.camera_to_detection_times),
            '📈 detection rate': calculate_rate(self.sends, total_send_attempts),
            '📉 error rate': calculate_rate(total_errors, total_send_attempts),
        }

    def get_errors(self):
        return self.logs

    def get_client_data(self, client_ip: Optional[str] = None):
        if client_ip:
            # Find the full key that contains the given IP
            matched_key = next(
                (key for key in self.clients_track if client_ip in key), None)

            if matched_key:
                # Sort and return the data for the found key
                return {matched_key: dict(sorted(self.clients_track[matched_key].items()))}
            else:
                return "No data available"

        # Returns all data with each client's cameras sorted by ID
        return {ip: dict(sorted(channels.items())) for ip, channels in self.clients_track.items()}

    async def reset_metrics(self, delete_params: str = "All"):
        async with self._metrics_lock:
            if delete_params != "Clients":
                if delete_params == "All":
                    self.start_time = datetime.now()
                    self.errors = {'get': 0, 'send': 0,
                                   'delete': 0, 'general': 0}
                self.receives = 0
                self.sends = 0
                self.Alert_in_action = 0
                self.motion_mask_time.clear()
                self.no_motion = 0
                self.no_detection = 0
                self.no_detection_on_mask = 0
                self.expires = 0
                self.clients_track.clear()
                self.processing_times.clear()
                self.camera_to_detection_times.clear()
            else:
                self.clients_track.clear()


metrics_tracker = MetricsTracker()
