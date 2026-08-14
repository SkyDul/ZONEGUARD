# ZONEGUARD – Sistem Deteksi Intrusi Real-Time

ZONEGUARD adalah aplikasi web berbasis *Computer Vision* untuk mendeteksi penyusup (intruder) pada area terlarang secara real-time. Aplikasi ini dirancang agar pengguna dapat menggambar "zona terlarang" langsung di tampilan kamera, dan AI akan otomatis membunyikan alarm jika ada penyusup yang masuk.

---

## 🧠 Penjelasan Konsep & Teknologi (Untuk Pemula)

Banyak yang bingung bagaimana sistem ini bekerja, model apa yang dipakai, dan apa itu "data training". Berikut adalah penjelasan detailnya dengan bahasa yang mudah dipahami.

### 1. Model AI yang Digunakan
Sistem ini menggunakan **YOLOv8 (You Only Look Once versi 8)**. YOLO adalah salah satu model pendeteksi objek terbaik dan tercepat di dunia saat ini.
- **Tugas Model:** Melihat sebuah gambar, lalu mengenali di mana lokasi objek dan apa nama objek tersebut (misalnya: "Ini orang, posisinya di kordinat X,Y").
- **Optimalisasi (OpenVINO):** Agar YOLOv8 bisa berjalan sangat cepat di laptop atau PC biasa (tanpa kartu grafis super mahal), sistem ini mengubah modelnya menggunakan teknologi **Intel OpenVINO (Quantization INT8)**. Ini membuat model berjalan sangat ringan di CPU Intel. Jika komputer menggunakan AMD atau perangkat lain, sistem akan otomatis menggunakan model dasar versi **PyTorch**.

### 2. Metode Pelacakan (Tracking) & Zona (Geofencing)
YOLOv8 hanya mendeteksi "Ada orang di frame ini". Tapi dia tidak tahu apakah orang di detik pertama adalah orang yang sama di detik kedua.
- **Metode Tracker:** Kita menggunakan metode **ByteTrack-lite (IoU-based Tracker)**. Algoritma ini mencocokkan kotak objek (Bounding Box) di detik sebelumnya dengan detik saat ini menggunakan metode *Intersection over Union* (IoU). Dengan ini, sistem tahu bahwa penyusup tersebut adalah orang yang sama (misalnya: Penyusup ID #1) dan sistem dapat menggambar jejak garis pergerakannya (trail).
- **Metode Geofencing:** Untuk mengetahui apakah orang masuk ke zona terlarang, sistem menggunakan algoritma **Point-in-Polygon (Ray-casting)** lewat OpenCV. Sistem akan mengecek kordinat ujung kaki dari si penyusup, jika kakinya menyentuh area poligon yang kita gambar, maka dianggap sebagai intrusi.

### 3. Apa Itu Data Training? Di mana Data Trainingnya?
AI tidak secara ajaib mengerti apa itu "Penyusup" atau "Senjata" (sistem ini melacak kelas `0: gun` dan `1: intruder`). AI ini harus "disekolahkan" terlebih dahulu.
- **Data Training:** Ini adalah kumpulan ribuan foto manusia dan senjata di berbagai kondisi (siang, malam, posisi duduk, berdiri, dll).
- **Proses Labeling:** Semua foto tersebut diberi tanda kotak secara manual oleh manusia dan diberi nama.
- **Proses Belajar:** Foto-foto berlabel tersebut dimasukkan ke program pelatih YOLO. Model akan belajar pola matematis dari bentuk orang/senjata selama berjam-jam.
- **Hasil Akhir:** Di project ini, **kita tidak lagi melakukan proses training**. Proses training sudah dilakukan di masa lalu/tempat lain, dan hasil "otak" AI-nya disimpan dalam bentuk file berekstensi `.pt` atau `.xml/.bin` di dalam folder `models/`. Jadi, aplikasi ini hanya tinggal **memakai (Inference)** otak AI yang sudah pintar tersebut.

---

## 🔄 Alur Kerja Sistem (Pipeline)

Lalu, bagaimana perjalanan data dari kamera hingga alarm berbunyi? Berikut adalah alurnya secara urut:

1. **Input Kamera (Frontend):** Browser kamu mengakses kamera (Webcam), mengambil gambar berupa *frame* video setiap beberapa milidetik.
2. **Kirim ke Server:** Gambar itu diubah ke format teks (Base64) lalu dikirim ke server belakang layar (Python/Flask) lewat API (`POST /detect`).
3. **Preprocessing:** Sebelum dilihat AI, gambar diubah ukurannya secara paksa menjadi bentuk persegi (640x640 pixel) agar sesuai dengan standar otak model AI (Letterbox resize).
4. **Deteksi (Inference):** Gambar dimasukkan ke model YOLOv8. AI akan berpikir sekian milidetik, lalu mengeluarkan hasil berupa kumpulan kotak (Bounding Box) beserta persentase keyakinan (Confidence Score).
5. **Postprocessing (NMS):** Terkadang AI melihat 1 orang tapi memberinya 3 kotak tumpang tindih. Sistem menggunakan algoritma NMS (*Non-Maximum Suppression*) untuk menghapus kotak berlebih dan menyisakan 1 kotak paling akurat.
6. **Tracking & Zona:** Kotak hasil deteksi masuk ke tracker untuk diberi ID unik, lalu titik kakinya dites apakah berada di dalam zona terlarang (`cv2.pointPolygonTest`).
7. **Peringatan & Log (Output):** Jika kaki berada di zona terlarang, server membalas dengan status "BAHAYA". Browser akan membunyikan alarm *beep*, membuat layar berkedip merah, dan mencatat foto penyusup tersebut ke dalam panel *Event Log*.

---

## 🛠️ Persyaratan Sistem

Pastikan hal-hal berikut sudah ter-install di komputer/laptop kamu:
- **Python** (versi 3.8 ke atas disarankan)
- Kamera Web (Webcam) aktif

## 🚀 Cara Menjalankan Aplikasi

1. **Siapkan Project:** Buka terminal/Command Prompt lalu arahkan ke folder ini.
2. **Install Library:** Jalankan perintah `pip install -r requirements.txt`.
3. **Jalankan Server:** Jalankan perintah `python app.py`.
4. **Buka Aplikasi:** Buka browser dan ketik `http://localhost:5000`. Izinkan akses kamera.

## 🎮 Cara Menggunakan UI (User Interface)

- **Gambar Zona (Shortcut: D):** Klik tombol untuk mulai menggambar titik kordinat zona terlarang di layar kamera.
- **Mulai Deteksi (Shortcut: S):** Sistem mulai menjalankan AI untuk mencari penyusup.
- **Reset Zona (Shortcut: R):** Hapus area dan buat yang baru.
- **Mute/Unmute Audio (Shortcut: M):** Mematikan atau menghidupkan alarm suara.

---
*Dikembangkan untuk eksperimen deteksi intrusi cerdas menggunakan AI & Computer Vision.*
