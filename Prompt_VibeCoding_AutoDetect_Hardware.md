# Prompt Tambahan untuk Vibe Coding: Auto-Detect Hardware & Model Fallback

Copy-paste prompt di bawah ini (bisa digabung dengan prompt web app sebelumnya, atau dikirim terpisah sebagai lanjutan):

---

Tambahkan fitur auto-detect hardware ke backend aplikasi deteksi intrusi ini, dengan spesifikasi berikut:

## Tujuan
Aplikasi harus bisa berjalan tanpa error di berbagai spek komputer:
- Prosesor Intel tanpa NPU (contoh: Intel Core i3 generasi 10)
- Prosesor Intel dengan NPU (contoh: Intel Core Ultra)
- Prosesor selain Intel (contoh: AMD)

## Logic yang Dibutuhkan

1. **Deteksi brand CPU** saat aplikasi pertama kali dijalankan (startup), menggunakan library `py-cpuinfo`
2. **Jika CPU adalah Intel:**
   - Load model dalam format OpenVINO dari folder `best_openvino_model/` (berisi file `.xml` dan `.bin`)
   - Cek device OpenVINO yang tersedia lewat `openvino.Core().available_devices`
   - Pilih device dengan prioritas: NPU (jika tersedia) > GPU (jika tersedia) > CPU
   - Jika proses load OpenVINO gagal karena alasan apapun (exception), otomatis fallback ke model `.pt` sebagai cadangan
3. **Jika CPU bukan Intel (misal AMD):**
   - Langsung load model `.pt` (format PyTorch) dari file `best.pt`
   - Gunakan device `cuda` jika GPU NVIDIA tersedia, jika tidak gunakan `cpu`
4. **Model hanya di-load sekali** saat server backend pertama kali start (bukan setiap request), supaya tidak lambat
5. **Tampilkan informasi hardware yang terdeteksi** di log/console server saat startup, contoh:
   ```
   CPU terdeteksi: Intel(R) Core(TM) i3-10100
   Brand: Intel
   OpenVINO device tersedia: ['CPU']
   Model digunakan: OpenVINO (best_openvino_model), device: CPU
   ```
6. **(Opsional) Tampilkan info ini juga di frontend**, misal badge kecil di pojok halaman: "Running on: Intel CPU (OpenVINO)" atau "Running on: AMD (PyTorch)" — supaya user tahu mode yang sedang aktif

## Struktur File yang Perlu Disiapkan

```
project/
├── best.pt                      # model PyTorch, untuk non-Intel / fallback
├── best_openvino_model/         # model OpenVINO, untuk Intel
│   ├── best.xml
│   └── best.bin
├── model_loader.py               # modul khusus berisi logic deteksi hardware & load model
├── app.py                        # aplikasi utama, import fungsi dari model_loader.py
```

## Dependencies yang Perlu Ditambahkan

```
py-cpuinfo
openvino
ultralytics
torch
```

## Penanganan Error

- Semua proses deteksi hardware dan load model harus dibungkus try-except
- Jika kedua model (OpenVINO maupun .pt) sama-sama gagal di-load, tampilkan pesan error yang jelas ke user (bukan crash tanpa keterangan), misal: "Model tidak dapat dimuat. Pastikan file best.pt atau best_openvino_model tersedia di folder project."

---

**Catatan tambahan saat pakai prompt ini:**
- Sebutkan ke AI coding tool kamu path folder model yang sebenarnya kalau berbeda dari contoh di atas
- Kalau kamu juga pakai fitur tracking (ByteTrack/BoT-SORT) dari prompt sebelumnya, pastikan minta supaya tracker tetap konsisten dipakai baik saat model OpenVINO maupun .pt yang aktif
