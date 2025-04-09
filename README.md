# Face Detection

Aplikasi berbasis website untuk deteksi wajah secara real-time menggunakan FastAPI, OpenCV, dan DeepFace. Aplikasi ini memungkinkan pengguna untuk melakukan deteksi wajah melalui gambar yang diunggah atau wajah yang dideteksi melalui facecam atau webcam.

## ✨ Fitur

- ✅ Deteksi wajah secara real-time menggunakan webcam
- ✅ Unggah gambar(Belum terimplementasi) dan deteksi wajah
- ✅ Backend menggunakan FastAPI
- ✅ Kotak hijau muncul di sekitar wajah yang terdeteksi

## Daftar Requirement
- Python 3.9
- pip

## 🧾 Struktur Proyek

```
├── app.py              # Aplikasi utama
├── templates/           # Template HTML (Jinja2)
├── static/              # Aset statis
├── dataset/             # Folder dataset (dataset untuk pengenalan wajah)
|-> dataset/[nama_orang]/[nama_etnis]/[file.jpeg] # Tempat file dataset berada
├── requirements.txt     # Daftar dependensi Python yang digunakan
└── README.md            # File dokumentasi
```

## ⚙️ Cara Menjalankan

1. **Clone repository**
   ```bash
   git clone https://github.com/IlhamFaisalR23/pcd_facedetection.git
   cd pcd_facedetection
   ```

2. **Membuat dan mengaktifkan Virtual Environment**
   ```bash
   python -m venv venv
   source venv/scripts/activate        # Untuk macOS / Linux / Windows yang menggunakan Bash
   .\venv\Scripts\activate             # Untuk Windows
   ```

3. **Install dependensi**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan server**
   ```bash
   uvicorn app:app
   ```

5. **Buka di browser**
   ```
   http://localhost:8000
   ```

## 🧪 Cara Menggunakan

- Ikuti alur aplikasi yang tampil pada browser
- Memasukkan nama
- Memilih menu antara ingin mengunggah gambar (Belum terimplementasi)
- Atau memilih menu camera (Sudah terimplementasi sebagian)

## 🛠 Teknologi yang Digunakan

- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenCV](https://opencv.org/)
- [DeepFace](https://github.com/serengil/deepface)

