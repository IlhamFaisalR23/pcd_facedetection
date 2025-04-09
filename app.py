import os
import cv2
import numpy as np
from deepface import DeepFace
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from typing import List, Dict
from PIL import Image
import io
import logging
import threading
import time
from collections import defaultdict

# Config
DATASET_PATH = "dataset"
FACE_DETECTOR = "opencv"
MODEL_NAME = "VGG-Face"

def iterate_dataset_images():
    for name in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, name)
        if os.path.isdir(person_path):
            for root, _, files in os.walk(person_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        ethnic = os.path.basename(root)
                        yield name, ethnic, os.path.join(root, file)

def prepare_dataset():
    """Preprocess all dataset images into embeddings with full paths"""
    for name, ethnic, img_path in iterate_dataset_images():
        try:
            rep = DeepFace.represent(
                img_path=img_path,
                model_name=MODEL_NAME,
                detector_backend=FACE_DETECTOR,
                enforce_detection=False
            )[0]["embedding"]
            
            cached_embeddings.append({
                "name": name,
                "ethnic": ethnic,
                "embedding": rep,
                "path": img_path  # Store full path
            })
        except Exception as e:
            print(f"[ERROR] Failed embedding {img_path}: {e}")

def match_face_cached(face_img: np.ndarray) -> Dict:
    try:
        # Save temporary image
        temp_file = "temp_face.jpg"
        cv2.imwrite(temp_file, face_img)
        
        # Use DeepFace.find which handles everything internally
        dfs = DeepFace.find(
            img_path=temp_file,
            db_path=DATASET_PATH,
            model_name=MODEL_NAME,
            detector_backend=FACE_DETECTOR,
            enforce_detection=False,
            silent=True
        )
        
        os.remove(temp_file)  # Clean up
        
        if len(dfs) > 0 and len(dfs[0]) > 0:
            best_match = dfs[0].iloc[0]
            identity = best_match['identity']
            name = os.path.basename(os.path.dirname(identity))
            ethnic = os.path.basename(os.path.dirname(os.path.dirname(identity)))
            
            return {
                # dibalik
                "name": ethnic,
                "ethnic": name,
                "similarity": 1 - best_match['distance']
            }
    except :
        pass
    return None

# Global mapping nama → etnis
cached_embeddings = []
app = FastAPI()
prepare_dataset()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Face detector
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

name_to_ethnic = {}



# Di bagian awal (setelah imports), inisialisasi mapping ini:
def build_name_to_ethnic_map():
    global name_to_ethnic
    for name in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, name)
        if os.path.isdir(person_path):
            for ethnic in os.listdir(person_path):
                name_to_ethnic[name] = ethnic
                break  # karena 1 nama = 1 ras

build_name_to_ethnic_map()
# # Setup logging
# logging.basicConfig(level=logging.INFO)

def analyze_emotion(face_img: np.ndarray) -> str:
    try:
        analysis = DeepFace.analyze(
            face_img, 
            actions=['emotion'],
            detector_backend=FACE_DETECTOR,
            enforce_detection=False
        )
        return analysis[0]['dominant_emotion']
    except Exception as e:
        logging.error("Emotion analysis failed", exc_info=True)
        return "Unknown"

def match_face(face_img: np.ndarray) -> Dict:
    similarity_results = []
    for name, ethnic, img_path in iterate_dataset_images():
        try:
            result = DeepFace.verify(
                face_img, 
                img_path, 
                model_name=MODEL_NAME,
                detector_backend=FACE_DETECTOR,
                enforce_detection=False
            )
            if result["verified"]:
                similarity_results.append({
                    "name": name,
                    "similarity": result["distance"]
                })
        except Exception:
            continue
    
    if similarity_results:
        best = min(similarity_results, key=lambda x: x["similarity"])
        best["ethnic"] = name_to_ethnic.get(best["name"], "Unknown")
        return best
    return None


def detect_faces(image: np.ndarray) -> List[Dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    results = []
    for (x, y, w, h) in faces:
        face_img = image[y:y+h, x:x+w]
        emotion = analyze_emotion(face_img)
        best_match = match_face_cached(face_img)

        if best_match:
            similarity_percent = (1 - best_match["similarity"]) * 100
            summary = (
                f"Kamu terlihat seperti {best_match['name']} "
                f"(Etnis: {best_match['ethnic']}, "
                f"Confidence: {similarity_percent:.1f}%), "
                f"kelihatan {emotion}."
            )
            ethnic_result = best_match["ethnic"]
        else:
            summary = "Tidak ada yang mirip dalam dataset."
            ethnic_result = "Tidak diketahui"

        results.append({
            "box": (x, y, w, h),
            "emotion": emotion,
            "ethnic": ethnic_result,
            "best_match": best_match,
            "summary": summary
        })
    return results

def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    for det in detections:
        x, y, w, h = det["box"]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        info = [
            f"Etnis: {det.get('ethnic', '?')}",
            f"Emosi: {det.get('emotion', '?')}",
            det.get("summary", "")
        ]
        y_text = y - 10 if y - 10 > 10 else y + h + 10
        for i, line in enumerate(info):
            cv2.putText(image, line, (x, y_text + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
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

# ---------- Webcam Streaming -----------

def generate_frames():
    cap = cv2.VideoCapture(0)
    frame_lock = threading.Lock()
    current_frame = None
    cached_results = []

    def detect_and_analyze():
        nonlocal cached_results
        while True:
            with frame_lock:
                if current_frame is None:
                    continue
                frame_copy = current_frame.copy()

            gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            results = []
            for (x, y, w, h) in faces:
                face_img = frame_copy[y:y+h, x:x+w]
                emotion = analyze_emotion(face_img)
                best_match = match_face_cached(face_img)

                if best_match:
                    similarity_percent = (1 - best_match["similarity"]) * 100
                    summary = (
                        f"You look like {best_match['name']} "
                        f"(Ethnic: {best_match['ethnic']}, "
                        f"Confidence: {similarity_percent:.1f}%), "
                        f"appearing {emotion}."
                    )
                    ethnic_result = best_match["ethnic"]
                else:
                    summary = "No match found in dataset."
                    ethnic_result = "Unknown"

                results.append({
                    "box": (x, y, w, h),
                    "emotion": emotion,
                    "ethnic": ethnic_result,
                    "summary": summary
                })

            cached_results = results
            time.sleep(1.0)  # Deteksi ulang tiap 1 detik

    # Start background thread
    threading.Thread(target=detect_and_analyze, daemon=True).start()

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

            final_detections = []
            for (x, y, w, h) in faces:
                match = next(
                    (r for r in cached_results if abs(r["box"][0] - x) < 50 and abs(r["box"][1] - y) < 50),
                    None
                )
                if match:
                    final_detections.append({
                        "box": (x, y, w, h),
                        "emotion": match["emotion"],
                        "ethnic": match["ethnic"],
                        "summary": match["summary"]
                    })
                else:
                    final_detections.append({
                        "box": (x, y, w, h),
                        "emotion": "?",
                        "ethnic": "?",
                        "summary": "Detecting..."
                    })

            # Draw all info
            frame_with_box = draw_detections(frame.copy(), final_detections)
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
