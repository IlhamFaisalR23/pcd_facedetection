import os
import cv2
import numpy as np
import dlib
import random

INPUT_DIR = "dataset"
OUTPUT_DIR = "preprocessed_dataset"

# Detektor wajah dan landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_eye_centers(landmarks):
    left_eye_pts = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(36, 42)])
    right_eye_pts = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(42, 48)])
    left_eye_center = left_eye_pts.mean(axis=0).astype("int")
    right_eye_center = right_eye_pts.mean(axis=0).astype("int")
    return left_eye_center, right_eye_center

def align_face(image, left_eye_center, right_eye_center):
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx))
    eyes_center = tuple(((left_eye_center + right_eye_center) // 2).tolist())
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1)
    aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
    return aligned

def augment_image(image):
    augmented = []

    # Rotasi
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((112, 112), angle, 1)
    rotated = cv2.warpAffine(image, M, (224, 224))
    augmented.append(rotated)

    # Cermin horizontal
    flipped = cv2.flip(image, 1)
    augmented.append(flipped)

    # Brightness dan Contrast
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-30, 30)
    bright_contrast = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    augmented.append(bright_contrast)

    # Gaussian noise
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    augmented.append(noisy)

    return augmented

def preprocess_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        print(f"[x] Tidak bisa membaca gambar: {img_path}")
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    if len(rects) == 0:
        print(f"[x] Tidak ditemukan wajah: {img_path}")
        return []

    rect = rects[0]
    shape = predictor(gray, rect)

    try:
        left_eye, right_eye = get_eye_centers(shape)
        aligned = align_face(image, left_eye, right_eye)
    except Exception as e:
        print(f"[x] Gagal align wajah ({img_path}): {e}")
        return []

    (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
    x, y = max(x, 0), max(y, 0)
    cropped = aligned[y:y+h, x:x+w]
    resized = cv2.resize(cropped, (224, 224))

    ycrcb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    normalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    augmented = augment_image(normalized)
    return [normalized] + augmented

def process_dataset():
    for name in os.listdir(INPUT_DIR):
        person_path = os.path.join(INPUT_DIR, name)
        if not os.path.isdir(person_path):
            continue

        for ethnic in os.listdir(person_path):
            ethnic_path = os.path.join(person_path, ethnic)
            if not os.path.isdir(ethnic_path):
                continue

            for img_file in os.listdir(ethnic_path):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                input_path = os.path.join(ethnic_path, img_file)
                output_dir = os.path.join(OUTPUT_DIR, name, ethnic)
                os.makedirs(output_dir, exist_ok=True)

                results = preprocess_image(input_path)
                if results:
                    for i, img in enumerate(results):
                        fname = f"{os.path.splitext(img_file)[0]}_aug{i}.jpg"
                        output_path = os.path.join(output_dir, fname)
                        cv2.imwrite(output_path, img)
                    print(f"[✓] {input_path} -> {len(results)} file")
                else:
                    print(f"[x] Gagal preprocess: {input_path}")

if __name__ == "__main__":
    process_dataset()
