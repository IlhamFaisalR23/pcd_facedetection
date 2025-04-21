import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from typing import List, Dict
import io
import threading
import time
import base64
from deepface import DeepFace
import joblib
import tempfile

# Config
DATASET_PATH = "dataset"
MODEL_PATH = "dnn_models"

# Setup global
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

os.makedirs(MODEL_PATH, exist_ok=True)

prototxt_path = os.path.join(MODEL_PATH, "deploy.prototxt")
model_path = os.path.join(MODEL_PATH, "res10_300x300_ssd_iter_140000.caffemodel")

if not (os.path.exists(prototxt_path) and os.path.exists(model_path)):
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    import urllib.request
    print("Downloading model files...")
    urllib.request.urlretrieve(prototxt_url, prototxt_path)
    urllib.request.urlretrieve(model_url, model_path)
    print("Model files downloaded successfully.")


net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Detector untuk wajah
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces_dnn(image: np.ndarray, conf_threshold=0.5) -> List[Dict]:
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()
    
    results = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence < conf_threshold:
            continue
            
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        x, y = x1, y1
        w, h = x2 - x1, y2 - y1
        
        if w > 0 and h > 0:
            results.append({
                "box": (x, y, w, h),
                "confidence": float(confidence)
            })
    
    return results

def draw_detections_dnn(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    for det in detections:
        x, y, w, h = det["box"]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        confidence = det.get("confidence", 0)
        if confidence > 0:
            label = f"{confidence:.2f}"
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def crop_face(image: np.ndarray, detection: Dict, padding=0.2) -> np.ndarray:
    """Crop the face from the image with some padding"""
    if not detection:
        return None
    
    x, y, w, h = detection["box"]
    
    padding_x = int(w * padding)
    padding_y = int(h * padding)
    
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(image.shape[1], x + w + padding_x)
    y2 = min(image.shape[0], y + h + padding_y)

    face_crop = image[y1:y2, x1:x2]
    return face_crop

def detect_faces(image: np.ndarray) -> List[Dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    results = []
    for (x, y, w, h) in faces:
        results.append({"box": (x, y, w, h)})
    return results

def predict_ethnics(img_path, model_filename="my_model.pkl"):
    MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_filename)
    
    if not os.path.exists(MODEL):
        print(f"Model file {MODEL} not found!")
        return "Model Not Found", {}
    
    knn = joblib.load(MODEL)
    embedding = DeepFace.represent(
        img_path=img_path,
        model_name="VGG-Face",
        enforce_detection=False
    )[0]["embedding"]

    # Prediksi kelas
    pred_class = knn.predict([embedding])[0]
    
    # Prediksi probabilitas semua kelas
    probas = knn.predict_proba([embedding])[0]
    class_labels = knn.classes_

    # Print hasil
    print("Etnis terdeteksi:", pred_class)
    for label, prob in zip(class_labels, probas):
        print(f"{label}: {prob:.2f}")

    return pred_class, dict(zip(class_labels, probas))

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
    
    detections = detect_faces_dnn(image)
    response_data = {}
    
    if detections:
        best_detection = max(detections, key=lambda x: x.get("confidence", 0))
        result_image = draw_detections_dnn(image.copy(), detections)
        _, img_encoded = cv2.imencode('.jpg', result_image)
        result_img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
        
        face_crop = crop_face(image, best_detection)
        if face_crop is not None:
            _, crop_encoded = cv2.imencode('.jpg', face_crop)
            crop_img_base64 = base64.b64encode(crop_encoded.tobytes()).decode('utf-8')

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                tmp_filename = tmp_file.name
                cv2.imwrite(tmp_filename, face_crop)

            try:
                pred_class, _ = predict_ethnics(tmp_filename)
            except Exception as e:
                pred_class = "Tidak Diketahui"
            finally:
                os.remove(tmp_filename)

            response_data = {
                "status": "success",
                "result_image": f"data:image/jpeg;base64,{result_img_base64}",
                "face_crop": f"data:image/jpeg;base64,{crop_img_base64}",
                "race": pred_class
            }
        else:
            response_data = {
                "status": "error",
                "message": "Failed to crop face",
                "result_image": f"data:image/jpeg;base64,{result_img_base64}"
            }
    else:
        _, img_encoded = cv2.imencode('.jpg', image)
        result_img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
        
        response_data = {
            "status": "error",
            "message": "No faces detected",
            "result_image": f"data:image/jpeg;base64,{result_img_base64}"
        }
    
    return JSONResponse(content=response_data)

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