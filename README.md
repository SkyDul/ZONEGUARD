# ZONEGUARD – Sistem Deteksi Intrusi Real-Time

ZONEGUARD adalah aplikasi web berbasis Computer Vision untuk mendeteksi penyusup (intruder) pada area terlarang secara real-time. Aplikasi ini memanfaatkan **YOLOv8** dengan inferensi menggunakan **OpenVINO** untuk performa tinggi pada CPU, dilengkapi dengan fitur multi-object tracking (ByteTrack) dan manajemen zona dinamis yang bisa diatur langsung melalui antarmuka web.

---

## 🌟 Fitur Utama

- **Live Webcam Feed**: Menangkap input kamera secara langsung melalui browser (tanpa perlu install software kamera khusus).
- **Interactive Zone Mapping**: Menggambar area terlarang (polygon) sesuka hati cukup dengan klik-klik titik di atas tampilan video.
- **Real-Time Detection & Tracking**: Deteksi objek menggunakan YOLOv8 (dengan OpenVINO) dengan object tracking ID persisten per objek.
- **Movement Trail**: Fitur unik untuk menggambar jejak (trail) posisi pergerakan penyusup di kanvas.
- **Intrusion Alerts**: Peringatan visual (bounding box merah, banner menyala) dan peringatan audio (*beep* saat intrusi masuk zona).
- **Event Logging**: Rekaman otomatis berupa snapshot, timestamp, ID penyusup yang bisa dilihat langsung dari panel sidebar.

---

## 🛠️ Persyaratan Sistem

Pastikan hal-hal berikut sudah ter-install di komputer/laptop kamu:
- **Python** (versi 3.8 ke atas disarankan)
- Kamera Web (Webcam) aktif

---

## 🚀 Cara Menjalankan Aplikasi

Ikuti panduan berikut agar aplikasi bisa berjalan dengan lancar:

### 1. Clone / Siapkan Project
Buka terminal (Command Prompt / PowerShell) lalu arahkan ke folder ini:
```bash
cd E:\laragon\www\ZONEGUARD
# (Sesuaikan path dengan lokasi project kamu)
```

### 2. Install Dependencies (Library Python)
Jalankan perintah ini di dalam folder ZONEGUARD:
```bash
pip install -r requirements.txt
```
*(Proses ini akan menginstal `Flask`, `opencv-python`, `openvino`, dll)*

### 3. Jalankan Server Backend
Setelah proses instalasi library selesai, nyalakan server Flask dengan:
```bash
python app.py
```
Tunggu beberapa detik hingga muncul pesan di terminal: `✅ OpenVINO model loaded successfully.` dan `* Running on http://127.0.0.1:5000`.

### 4. Buka Aplikasi di Browser
Buka browser modern kesukaanmu (Google Chrome, Edge, dll), lalu ketik:
```
http://localhost:5000
```
> **Catatan Penting**: Saat pertama kali membuka, browser akan meminta izin (permission) untuk mengakses kamera. Silakan pilih **Allow / Izinkan**.

---

## 🎮 Cara Menggunakan UI (User Interface)

1. **Aktifkan Webcam**
   - Klik tombol biru `Aktifkan Webcam` di panel sebelah kiri. Video dari kamera kamu akan muncul di layar tengah.

2. **Membuat Zona Terlarang**
   - Klik tombol peringatan berwarna kuning `Gambar Zona`.
   - Di atas video kamera, klik pada beberapa titik untuk membentuk batas ruang (polygon).
   - Jika sudah dirasa pas, klik tombol `Tutup Polygon` di kiri. (Bisa klik `Reset Zona` untuk menghapus dan menggambar ulang).

3. **Mulai Deteksi**
   - Klik tombol hijau `Mulai Deteksi`.
   - Sistem akan langsung menandai *person* / *intruder* dengan kotak pelacak (bounding box) beserta jejak pergerakannya.
   - Jika orang tersebut masuk ke dalam zona yang sudah kamu gambar tadi, alarm akan berbunyi dan banner tanda bahaya akan menyala. Kejadian tersebut otomatis difoto masuk ke riwayat Log di panel sebelah kanan.

### ⌨️ Keyboard Shortcuts
- `D` : Mulai gambar zona / Tutup polygon
- `R` : Reset (hapus) zona
- `S` : Mulai / Hentikan deteksi
- `M` : Mute / Unmute audio alert
- `Esc` : Batal menggambar zona

---

## 📂 Struktur Direktori

```
ZONEGUARD/
├── app.py                    # Server backend Flask + Logic AI & OpenCV
├── tracker.py                # Modul pelacakan objek (IoU tracker / ByteTrack-lite)
├── requirements.txt          # Daftar package python
├── README.md                 # Dokumentasi (file yang sedang dibaca)
├── models/
│   └── best_openvino_model/  # Folder berisi hasil convert model YOLOv8 ke OpenVINO (.bin, .xml)
├── static/
│   ├── css/style.css         # Styling antarmuka web modern (Dark Theme)
│   ├── js/app.js             # Logic frontend untuk video, canvas, dan panggilan API
│   └── audio/alert.wav       # Audio peringatan intrusi
└── templates/
    └── index.html            # Kerangka dasar HTML dari aplikasi web
```

---

*Dikembangkan untuk eksperimen deteksi zona menggunakan Computer Vision.* Selamat mencoba!
