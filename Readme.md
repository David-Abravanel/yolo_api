# 🚀 YOLO API Service

A simple service to process images using **YOLO** object detection.

---

## 📦 Getting Started

### 1. Clone the Project

```bash
git clone https://github.com/David-Abravanel/yolo_api.git
cd yolo_api
```

### 2. Set up a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Service

```bash
python main.py
```

The service will start locally at `http://localhost:8000`.

---

## 🔧 Request Example

You can test the API using tools like [Postman](https://www.postman.com/) or `curl`.

### 🖥️ Endpoint

- **Method**: `POST`
- **URL**: `http://localhost:8000/yolo-detect`
- **Request Body (JSON)**:

```json
{
  "ip": "192.168.1.10",
  "nvr_name": "NVR_01",
  "channel_id": "01",
  "event_time": "2025-01-24T12:45:30",
  "event_type": "motion_detected",
  "snapshots": [
    "https://example.com/snapshot1.jpg",
    "https://example.com/snapshot2.jpg"
  ],
  "camera_data": {
    "confidence": 0.5,
    "classes": [0, 2],
    "is_focus": false,
    "masks": [
      {
        "shape": [
          { "x": 0.1234, "y": 0.2134 },
          { "x": 0.9876, "y": 0.5543 },
          { "x": 0.1212, "y": 0.3333 }
        ]
      },
      {
        "shape": [
          { "x": 0.5432, "y": 0.6543 },
          { "x": 0.8765, "y": 0.1234 },
          { "x": 0.4321, "y": 0.7654 }
        ]
      }
    ]
  }
}
```

### 📊 Explanation of Parameters

#### `camera_data`

- **`confidence`** (`float`): Minimum confidence threshold for detection. Default: `0.2`.
- **`classes`** (`list[int]`): Object classes to detect. Default: `[0, 1, 2]` (e.g., person, bicycle, car).
- **`is_focus`** (`bool`): If `true`, the detection focuses on the areas defined by the masks. Default: `true`.
- **`masks`** (`list`): List of polygon shapes to define the focus areas. Default: an empty list.

#### Example Mask

Each mask is a list of `x, y` coordinates representing the vertices of a polygon.

---

## 💡 Example Response

```json
{
  "camera_data": {
    "ip": "192.168.1.10",
    "nvr_name": "NVR_01",
    "channel_id": "01",
    "event_time": "2025-01-24T12:45:30",
    "event_type": "motion_detected",
    "snapshots": [
      "https://example.com/snapshot1.jpg",
      "https://example.com/snapshot2.jpg"
    ],
    "time_detect": "2025-01-24T12:45:45"
  },
  "detections": [
    [
      {
        "bbox": { "x1": 0.156, "y1": 0.224, "x2": 0.432, "y2": 0.666 },
        "class_name": "person",
        "class_id": 0,
        "confidence": 0.85
      },
      {
        "bbox": { "x1": 0.554, "y1": 0.372, "x2": 0.812, "y2": 0.734 },
        "class_name": "car",
        "class_id": 2,
        "confidence": 0.78
      }
    ],
    [
      {
        "bbox": { "x1": 0.156, "y1": 0.224, "x2": 0.432, "y2": 0.666 },
        "class_name": "person",
        "class_id": 0,
        "confidence": 0.85
      },
      {
        "bbox": { "x1": 0.554, "y1": 0.372, "x2": 0.812, "y2": 0.734 },
        "class_name": "car",
        "class_id": 2,
        "confidence": 0.78
      }
    ]
  ]
}
```

### 📝 Explanation of Response

- **`camera_data`**: Contains the time of detection (`time_detect`) after the request has been processed, without the original camera data.
- **`detections`**: A list of detected objects, each containing:
  - **`class_name`**: The name of the detected object (e.g., "person", "car").
  - **`confidence`**: The confidence level of the detection.
  - **`bbox`**: The bounding box for the detected object in normalized coordinates (`x1`, `y1`, `x2`, `y2`).

---

## 🔧 AlertsRequest and AlertsResponse

### **AlertsRequest**

This model represents the incoming request to the API.

```python
class AlertsRequest(BaseModel):
    ip: str
    nvr_name: str
    channel_id: str
    event_time: datetime
    event_type: str
    snapshots: List[str]
    camera_data: CameraData = CameraData()

    def without_camera_data(self):
        # Create a dictionary without `camera_data` and add `time_detect`
        data = self.dict(exclude={"camera_data"})
        data["time_detect"] = str(datetime.now())  # Add current time
        return data
```

- The `without_camera_data()` function removes the `camera_data` from the request and adds the `time_detect` parameter with the current time.

### **AlertsResponse**

This model represents the response from the API. It contains the time of detection and a list of detections.

```python
class AlertsResponse(BaseModel):
    camera_data: Request
    detections: List[List[Detection]] | List
```

---

## 🚨 Troubleshooting

If the service doesn't start or behaves unexpectedly, ensure:

- You are using the correct Python version (e.g., Python 3.8+).
- All dependencies are installed correctly.
- The `url` in the request is valid and accessible.

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## 📦 PROD Commands

### Basic libs

```bash
sudo apt update && sudo apt upgrade -y && sudo apt install -y python3 python3-venv python3-pip git && sudo apt install -y libgl1
```

### Clone the Repo and Open the Folder

```bash
git clone https://github.com/David-Abravanel/yolo_api.git
cd yolo_api
```

### Open Virtual Environment and Activate

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Requirements

```bash
pip install fastapi uvicorn gunicorn ultralytics opencv-python
pip install aiohttp aioboto3
```

### Test

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## ⚙️ Systemd Background Run

### Open a New File

```bash
sudo nano /etc/systemd/system/yolo_api.service
```

### Copy and Paste the Systemd yolo_api File

```ini
[Unit]
Description=FastAPI application with Uvicorn YOLO SQS S3 For object detection
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/yolo_api
ExecStart=/bin/bash -c 'source /home/ubuntu/yolo_api/venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4'
Restart=always
RestartSec=3
TimeoutSec=30
Environment=PATH=/home/ubuntu/yolo_api/venv/bin:/usr/bin:/bin
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

### Upload the Changes and Start

```bash
sudo systemctl daemon-reload
sudo systemctl enable yolo_api.service
sudo systemctl start yolo_api.service
```

### Check Status of the Runner

```bash
sudo systemctl status yolo_api.service
```

### See the Run in Real-Time

```bash
sudo journalctl -u yolo_api.service -f
```

```bash
ssh -i C:\project\ILG\yolo_api\keys\ILG-YOLO-KP.pem ubuntu@51.17.47.54
```
