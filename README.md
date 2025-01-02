# Sistem Pakar Pendeteksi Penyakit Pneumonia Menggunakan Algoritma CNN

## Deskripsi Proyek
Proyek ini adalah aplikasi berbasis web yang dirancang untuk mendeteksi penyakit pneumonia dari gambar X-ray paru-paru menggunakan algoritma Convolutional Neural Network (CNN). Aplikasi ini dibuat untuk membantu masyarakat umum maupun tenaga medis dalam melakukan deteksi dini pneumonia dengan mudah dan cepat.

Aplikasi ini terdiri dari beberapa fitur utama seperti login, prediksi, informasi tentang pneumonia, dan visualisasi performa model. Model yang digunakan telah dilatih pada dataset pneumonia dan mampu membedakan antara gambar X-ray yang menunjukkan kondisi normal atau pneumonia dengan tingkat akurasi tinggi.

---

## Teknologi yang Digunakan
- **Python**: Bahasa pemrograman utama untuk membangun aplikasi.
- **Streamlit**: Framework untuk membuat antarmuka web interaktif.
- **TensorFlow/Keras**: Digunakan untuk memuat dan memproses model deep learning.
- **PIL (Pillow)**: Untuk memproses gambar X-ray yang diunggah pengguna.
- **NumPy**: Untuk manipulasi array dan data numerik.

---

## Fitur Aplikasi

### 1. **Login**
- Pengguna harus login terlebih dahulu untuk mengakses aplikasi.
- Default username dan password adalah:
  - **Username**: `admin`
  - **Password**: `admin123`
- Jika login berhasil, pengguna diarahkan ke halaman utama aplikasi.

### 2. **Halaman Home**
- Menampilkan deskripsi tentang aplikasi dan pentingnya deteksi dini pneumonia.
- Informasi tentang cara kerja aplikasi berbasis CNN.
- Visualisasi akurasi dan performa model selama pelatihan.

### 3. **Halaman Prediksi**
- Pengguna dapat mengunggah gambar X-ray paru-paru dalam format `jpg`, `jpeg`, atau `png`.
- Aplikasi akan memproses gambar menggunakan model CNN yang telah dilatih.
- Menampilkan hasil prediksi (Normal/Pneumonia) dengan tingkat kepercayaan (%).
- Memberikan saran berdasarkan hasil prediksi.

### 4. **Halaman Tentang**
- Informasi tentang pengembang aplikasi:
  - **Nama**: Fajar Pangestu Amandaru
  - **Universitas**: Universitas Indraprasta PGRI
- Media sosial dan email pengembang untuk kontak lebih lanjut.

### 5. **Sidebar Navigasi**
- Sidebar interaktif untuk memudahkan pengguna berpindah antar halaman:
  - **Home**: Mengarahkan ke halaman utama.
  - **Prediksi**: Mengarahkan ke halaman prediksi.
  - **About**: Mengarahkan ke halaman tentang pengembang.

---

## Cara Menggunakan
1. Jalankan aplikasi menggunakan perintah `streamlit run nama_file.py` di terminal.
2. Login dengan username dan password yang telah disediakan.
3. Gunakan sidebar untuk navigasi antara halaman.
4. Pada halaman Prediksi, unggah gambar X-ray paru-paru.
5. Tunggu hingga hasil prediksi ditampilkan beserta tingkat kepercayaannya.

---

## Struktur Folder
```
|-- image/                # Folder untuk gambar pendukung aplikasi
|   |-- baner_pneumonia.jpg
|   |-- tutor_tb.jpg
|   |-- model_akurasi_pneumonia.jpg
|   |-- acc_perclass.png
|   |-- confus_matrix.png
|   |-- foto_profil.jpg
|
|-- keras_model_pneumonia.h5  # File model CNN yang telah dilatih
|-- labels_pneumonia.txt      # File label untuk kelas prediksi
|-- main.py                  # Kode utama aplikasi
```

---

## Model yang Digunakan
Model deep learning berbasis CNN dilatih menggunakan dataset X-ray paru-paru dari Kaggle. Performa model adalah sebagai berikut:
- **Akurasi pelatihan**: >90%
- **Akurasi per kelas**:
  - Normal: 94%
  - Pneumonia: 98%
- **Confusion Matrix**: Menunjukkan prediksi benar dan salah untuk masing-masing kelas.

---

## Kontak Pengembang
- **Nama**: Fajar Pangestu Amandaru
- **Email**: [fajar.faps@gmail.com](mailto:fajar.faps@gmail.com)
- **LinkedIn**: [Fajar Pangestu Amandaru](https://www.linkedin.com/in/fajarpangestuamandaru/)
- **Instagram**: [fjrpangestu](https://www.instagram.com/fjrpangestu)

---

## Lisensi
Proyek ini dibuat untuk keperluan pendidikan dan tugas akhir. Penggunaan kode dan model dapat dilakukan untuk tujuan non-komersial dengan menyertakan atribusi kepada pengembang.

---

Terima kasih telah menggunakan aplikasi ini! Semoga bermanfaat dalam membantu deteksi dini pneumonia.

