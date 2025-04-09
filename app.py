import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from typing import List, Dict
import io
import threading
import time

# Config
DATASET_PATH = "dataset"

# Setup global
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Detector untuk wajah
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces(image: np.ndarray) -> List[Dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    results = []
    for (x, y, w, h) in faces:
        results.append({"box": (x, y, w, h)})
    return results


def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    for det in detections:
        x, y, w, h = det["box"]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/cam", response_class=HTMLResponse)
async def cam_page(request: Request):
    return templates.TemplateResponse("cam.html", {"request": request})


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    detections = detect_faces(image)
    result_image = draw_detections(image.copy(), detections)
    _, img_encoded = cv2.imencode('.jpg', result_image)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")


# ---------- Webcam -----------

def generate_frames():
    cap = cv2.VideoCapture(0)
    frame_lock = threading.Lock()
    current_frame = None

    def detect_loop():
        nonlocal current_frame
        while True:
            with frame_lock:
                if current_frame is None:
                    continue
                frame_copy = current_frame.copy()

            time.sleep(1.0)

    threading.Thread(target=detect_loop, daemon=True).start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (480, 240))
            with frame_lock:
                current_frame = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            detections = [{"box": (x, y, w, h)} for (x, y, w, h) in faces]
            frame_with_box = draw_detections(frame.copy(), detections)

            _, buffer = cv2.imencode('.jpg', frame_with_box)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()


@app.get("/webcam")
async def webcam_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


# ---------- Main ----------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)