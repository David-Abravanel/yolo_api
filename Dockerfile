# FROM python:3.12-slim

# RUN apt-get update && \
#     apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
#     rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# RUN python -m venv /opt/venv

# ENV PATH="/opt/venv/bin:$PATH"

# COPY requirements.txt .

# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# RUN pip install --no-cache-dir \
#     torch==2.6.0+cpu torchvision==0.21.0+cpu \
#     --index-url https://download.pytorch.org/whl/cpu

# RUN pip install --no-cache-dir ultralytics[all]

# COPY . .

# EXPOSE 8000

# CMD ["python", "main.py"]

# שימוש ב-Python 3.12-slim
FROM python:3.12-slim

# התקנת התלויות הדרושות
RUN apt-get update && \
    apt-get install -y git libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# הגדרת תיקיית העבודה
WORKDIR /app

# יצירת סביבה וירטואלית
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# שכפול הריפוזיטורי (פעם ראשונה בלבד)
RUN git clone https://github.com/David-Abaravanel/yolo_api.git /app/yolo_api

# התקנת התלויות הראשונות מ-requirements.txt (פעם אחת)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/yolo_api/requirements.txt

# התקנת ספריות כבדות (פעם אחת בבנייה בלבד)
RUN pip install --no-cache-dir \
    torch==2.6.0+cpu torchvision==0.21.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir ultralytics[all]

# חשיפת הפורט
EXPOSE 8000

# פקודת ההפעלה של הקונטיינר
CMD git -C /app/yolo_api pull && \
    pip install --no-cache-dir -r /app/yolo_api/requirements.txt && \
    exec gunicorn -w 4 -b 0.0.0.0:8000 main:app --timeout 0
