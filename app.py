import os
import cv2
import numpy as np
from collections import Counter
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
from deepface import DeepFace
import uvicorn
from typing import List, Dict
import io
import threading
import time

# Config
DATASET_PATH = "dataset"
SIMILARITY_MODEL_PATH = "face_similarity_model.pkl"
ETHNICS_MODEL_PATH = "ethnics_detection_model.pkl"
SIMILARITY_SVC_MODEL = None
ETHNICS_SVC_MODEL = None

# Setup global
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Detector untuk wajah
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_model():
    global SIMILARITY_SVC_MODEL, ETHNICS_SVC_MODEL
    if ETHNICS_SVC_MODEL is None and os.path.exists(ETHNICS_MODEL_PATH):
        ETHNICS_SVC_MODEL = joblib.load(ETHNICS_MODEL_PATH)
    if SIMILARITY_SVC_MODEL is None and os.path.exists(SIMILARITY_MODEL_PATH):
        SIMILARITY_SVC_MODEL = joblib.load(SIMILARITY_MODEL_PATH)
    return ETHNICS_SVC_MODEL, SIMILARITY_SVC_MODEL

def predict_face_identity(img_path: str) -> str:
    _, model = load_model()

    try:
        # Ambil embedding dari gambar input
        embedding = DeepFace.represent(
            img_path=img_path,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]

        # Ubah ke numpy array dan reshape untuk prediksi
        embedding = np.array(embedding).reshape(1, -1)

        # Prediksi nama orang
        prediction = model.predict(embedding)[0]
        probability = np.max(model.predict_proba(embedding))

        print(f"✅ Orang terdeteksi: {prediction} (Probabilitas: {probability:.2f})")
        return f"{prediction} ({probability:.2f})"
    except Exception as e:
        print(f"❌ Gagal mengenali wajah dari '{img_path}': {e}")
        return "Tidak Dikenali"

def predict_face_identity_camera(face):
    try:
        # Load model untuk face similarity (ambil hanya model face similarity saja)
        _, model = load_model()
        if model is None:
            return "Model belum tersedia"
        
        # Ekstrak embedding wajah dengan DeepFace dan model Facenet
        embedding = DeepFace.represent(
            img_path=face,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]

        # Prediksi identitas dari embedding
        pred = model.predict([embedding])[0]
        prob = np.max(model.predict_proba([embedding]))
        return f"{pred} ({prob:.2f})"
    
    except Exception as e:
        return "Tidak Dikenali"

def predict_ethnics(img_path, MODEL="my_model.pkl"):
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


def predict_ethnics_camera(face):
    try:
        # Load model (Ambil hanya model ethnics saja)
        model, _ = load_model()
        if model is None:
            return "Model belum tersedia"
        
        # Ekstrak embedding wajah dengan DeepFace
        embedding = DeepFace.represent(
            img_path=face, 
            model_name="VGG-Face", 
            enforce_detection=False
        )[0]["embedding"]

        # Prediksi etnis dari embedding
        pred = model.predict([embedding])[0]
        return pred
    except Exception as e:
        return "Tidak diketahui"
    

def detect_faces(image: np.ndarray) -> List[Dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    results = []
    for (x, y, w, h) in faces:
        results.append({"box": (x, y, w, h)})
    return results

frame_counter = 0
cached_results = []

def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    global frame_counter, cached_results
    frame_counter += 1

    new_results = []

    for det in detections:
        x, y, w, h = det["box"]
        face = image[y:y+h, x:x+w]

        # Hanya prediksi setiap 30 frame sekali
        if frame_counter % 30 == 0:
            etnis = predict_ethnics_camera(face)
            kemiripan = predict_face_identity_camera(face)
            new_results.append((x, y, w, h, etnis, kemiripan))
        else:
            # Gunakan hasil dari sebelumnya
            new_results = cached_results

    cached_results = new_results

    # Gambar hasil prediksi
    for (x, y, w, h, etnis, kemiripan) in cached_results:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{etnis} | {kemiripan}"
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
    cap = cv2.VideoCapture(0)  # Akses webcam
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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

            # Deteksi wajah dalam gambar
            detections = detect_faces(frame)
            
            # Gambar bounding box dan prediksi etnis
            frame_with_box = draw_detections(frame.copy(), detections)

            # Encode gambar menjadi JPEG untuk streaming
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