import os
import cv2
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
from deepface import DeepFace

DATASET_PATH = "dataset"

def train_face_similarity_model(DATASET_PATH, MODEL="face_similarity_model.pkl"):
    if os.path.exists(MODEL):
        print(f"Model sudah ada di {MODEL}, proses training dilewati.")
        return

    print("Memulai proses training model kemiripan wajah...\n")

    X = []  # embeddings
    y = []  # labels
    total_images = 0

    # Loop melalui dataset yang memiliki struktur nama/etnis/image.jpg
    for person_name in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person_name)

        if not os.path.isdir(person_path):
            continue

        print(f"Memproses data untuk: {person_name}")

        # Loop untuk setiap etnis dalam folder orang tersebut
        for ethnic_label in os.listdir(person_path):
            ethnic_path = os.path.join(person_path, ethnic_label)

            if not os.path.isdir(ethnic_path):
                continue

            # Loop melalui gambar di dalam folder etnis
            for img_name in os.listdir(ethnic_path):
                img_path = os.path.join(ethnic_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Dapatkan embedding wajah menggunakan model "Facenet"
                    embedding = DeepFace.represent(
                        img_path=img_path,
                        model_name="Facenet",
                        enforce_detection=False
                    )[0]["embedding"]

                    X.append(embedding)
                    y.append(person_name)  # Label adalah nama orang (identitas)
                    total_images += 1

                    if total_images % 10 == 0:
                        print(f"Gambar ke-{total_images} diproses...")

                except Exception as e:
                    print(f"Error pada {img_path}: {e}")

    print(f"\nTotal gambar yang berhasil diproses: {total_images}")

    # Hitung jumlah data per orang
    person_counts = Counter(y)
    print("\nJumlah data per orang:")
    for person, count in person_counts.items():
        print(f"- {person}: {count} gambar")

    if total_images == 0:
        print("Tidak ada gambar yang berhasil diproses. Pastikan struktur folder dataset benar.")
        return

    # Convert ke numpy array
    X = np.array(X)
    y = np.array(y)

    # Split data
    print("\nMelakukan split data untuk training dan testing...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Latih classifier SVC
    print("Melatih model SVC...")
    svc = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    svc.fit(X_train, y_train)

    # Evaluasi
    print("\nEvaluasi model:")
    y_pred = svc.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Simpan model
    joblib.dump(svc, MODEL)
    print(f"\n✅ Model berhasil disimpan di {MODEL}")


def train_ethnics_model(DATASET_PATH, MODEL="ethnics_detection_model.pkl"):
    if os.path.exists(MODEL):
        print(f"Model sudah ada di {MODEL}, proses training dilewati.")
        return

    print("Memulai proses training model deteksi etnis...\n")

    X = []  # embeddings
    y = []  # labels
    total_images = 0

    for person_name in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person_name)

        if os.path.isdir(person_path):
            print(f"Memproses data untuk: {person_name}")
            for ethnic_label in os.listdir(person_path):
                ethnic_path = os.path.join(person_path, ethnic_label)

                if not os.path.isdir(ethnic_path):
                    continue

                for img_name in os.listdir(ethnic_path):
                    img_path = os.path.join(ethnic_path, img_name)
                    try:
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        embedding = DeepFace.represent(
                            img_path=img_path,
                            model_name="VGG-Face",
                            enforce_detection=False
                        )[0]["embedding"]

                        X.append(embedding)
                        y.append(ethnic_label)
                        total_images += 1

                        if total_images % 10 == 0:
                            print(f"Gambar ke-{total_images} diproses...")

                    except Exception as e:
                        print(f"Error pada {img_path}: {e}")

    print(f"\nTotal gambar yang berhasil diproses: {total_images}")

    # Hitung jumlah data per etnis
    ethnic_counts = Counter(y)
    print("\nJumlah data per etnis:")
    for ethnic, count in ethnic_counts.items():
        print(f"- {ethnic}: {count} gambar")

    if total_images == 0:
        print("Tidak ada gambar yang berhasil diproses. Pastikan struktur folder dataset benar.")
        return

    # Convert ke numpy array
    X = np.array(X)
    y = np.array(y)

    # Split data
    print("\nMelakukan split data untuk training dan testing...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Latih classifier SVC
    print("Melatih model SVC...")
    svc = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    svc.fit(X_train, y_train)

    # Evaluasi
    print("\nEvaluasi model:")
    y_pred = svc.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Simpan model
    joblib.dump(svc, MODEL)
    print(f"\n✅ Model berhasil disimpan di {MODEL}")