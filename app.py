import os
import cv2
import numpy as np
from collections import Counter
from fastapi import FastAPI, UploadFile, Form, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
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
import base64
from deepface import DeepFace
import joblib
import tempfile

# Config
DATASET_PATH = "preprocessed_dataset"
SIMILARITY_MODEL_PATH = "face_similarity_model.pkl"
ETHNICS_MODEL_PATH = "ethnics_detection_model.pkl"
SIMILARITY_SVC_MODEL = None
ETHNICS_SVC_MODEL = None
MODEL_PATH = "dnn_models"

# Inisialisasi FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Pastikan folder model ada
os.makedirs(MODEL_PATH, exist_ok=True)

# Path untuk model deteksi wajah
prototxt_path = os.path.join(MODEL_PATH, "deploy.prototxt")
model_path = os.path.join(MODEL_PATH, "res10_300x300_ssd_iter_140000.caffemodel")

# Download model jika belum ada
if not (os.path.exists(prototxt_path) and os.path.exists(model_path)):
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    import urllib.request
    print("Downloading model files...")
    urllib.request.urlretrieve(prototxt_url, prototxt_path)
    urllib.request.urlretrieve(model_url, model_path)
    print("Model files downloaded successfully.")

# Load model deteksi wajah
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Detektor wajah alternatif
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

def predict_ethnics(img_path, MODEL="ethnics_detection_model.pkl"):
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

# Fungsi untuk deteksi wajah menggunakan DNN
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
        
        # Pastikan koordinat tidak keluar dari gambar
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

# Fungsi untuk menggambar kotak deteksi wajah
def draw_detections_dnn(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    for det in detections:
        x, y, w, h = det["box"]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        confidence = det.get("confidence", 0)
        if confidence > 0:
            label = f"{confidence:.2f}"
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Fungsi untuk memotong wajah dari gambar
def crop_face(image: np.ndarray, detection: Dict, padding=0.2) -> np.ndarray:
    if not detection:
        return None
    
    x, y, w, h = detection["box"]
    
    # Tambahkan padding di sekitar wajah
    padding_x = int(w * padding)
    padding_y = int(h * padding)
    
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(image.shape[1], x + w + padding_x)
    y2 = min(image.shape[0], y + h + padding_y)

    face_crop = image[y1:y2, x1:x2]
    return face_crop

# =====================================
# ROUTE UTAMA
# =====================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Halaman utama"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/cam", response_class=HTMLResponse)
async def cam_page(request: Request):
    """Halaman kamera"""
    return templates.TemplateResponse("cam.html", {"request": request})

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Halaman upload gambar"""
    return templates.TemplateResponse("upload.html", {"request": request})

# =====================================
# API UNTUK UPLOAD GAMBAR
# =====================================

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Endpoint untuk upload gambar dan deteksi wajah"""
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    
    detections = detect_faces_dnn(image)
    response_data = {}
    
    if detections:
        # Ambil wajah dengan confidence tertinggi
        best_detection = max(detections, key=lambda x: x.get("confidence", 0))
        result_image = draw_detections_dnn(image.copy(), detections)
        
        # Konversi gambar ke base64
        _, img_encoded = cv2.imencode('.jpg', result_image)
        result_img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
        
        # Crop wajah
        face_crop = crop_face(image, best_detection)
        if face_crop is not None:
            _, crop_encoded = cv2.imencode('.jpg', face_crop)
            crop_img_base64 = base64.b64encode(crop_encoded.tobytes()).decode('utf-8')

            # Simpan sementara untuk prediksi
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                tmp_filename = tmp_file.name
                cv2.imwrite(tmp_filename, face_crop)

            try:
                # Prediksi etnis
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
                "message": "Gagal memotong wajah",
                "result_image": f"data:image/jpeg;base64,{result_img_base64}"
            }
    else:
        # Jika tidak ada wajah terdeteksi
        _, img_encoded = cv2.imencode('.jpg', image)
        result_img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
        
        response_data = {
            "status": "error",
            "message": "Tidak ada wajah terdeteksi",
            "result_image": f"data:image/jpeg;base64,{result_img_base64}"
        }
    
    return JSONResponse(content=response_data)

# =====================================
# API UNTUK WEBCAM
# =====================================

def generate_frames():
    cap = cv2.VideoCapture(0)
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
    """Endpoint untuk streaming webcam"""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# =====================================
# API UNTUK UPLOAD DATASET
# =====================================

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...), nama: str = Form(...), ras: str = Form(...)):
    """Endpoint untuk upload dataset wajah"""
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    # Deteksi wajah
    detections = detect_faces_dnn(image)
    if not detections:
        return JSONResponse(content={"status": "error", "message": "Tidak ada wajah terdeteksi"})

    # Ambil wajah dengan confidence tertinggi
    best_detection = max(detections, key=lambda x: x.get("confidence", 0))
    face_crop = crop_face(image, best_detection)

    if face_crop is None:
        return JSONResponse(content={"status": "error", "message": "Gagal memotong wajah"})

    # Simpan gambar ke folder dataset
    os.makedirs(DATASET_PATH, exist_ok=True)
    nama_folder = os.path.join(DATASET_PATH, nama)
    os.makedirs(nama_folder, exist_ok=True)

    ras_folder = os.path.join(nama_folder, ras)
    os.makedirs(ras_folder, exist_ok=True)

    # Buat nama file unik
    filename = f"{int(time.time() * 1000)}.jpg"
    path = os.path.join(ras_folder, filename)
    cv2.imwrite(path, face_crop)

    return JSONResponse(content={
        "status": "success",
        "message": "File berhasil diupload dan wajah dipotong",
        "path": path
    })

# =====================================
# JALANKAN SERVER
# =====================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)