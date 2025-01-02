import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf

# Memuat model dan label dengan penanganan kesalahan
model_path = 'keras_model_pneumonia.h5'  # Path ke model
labels_path = 'labels_pneumonia.txt'  # Path ke label

try:
    # Workaround untuk argumen tidak dikenal di DepthwiseConv2D
    from tensorflow.keras.layers import DepthwiseConv2D

    class CustomDepthwiseConv2D(DepthwiseConv2D):
        def __init__(self, groups=1, **kwargs):
            kwargs.pop('groups', None)  # Hapus argumen 'groups' jika ada
            super().__init__(**kwargs)

    tf.keras.utils.get_custom_objects()['DepthwiseConv2D'] = CustomDepthwiseConv2D

    model = load_model(model_path)
    class_names = open(labels_path, "r").readlines()
except Exception as e:
    st.error(f"Error loading model or labels: {e}")
    model = None
    class_names = []

# Konfigurasi gambar
IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224

# Fungsi untuk prediksi
def predict(image, model):
    # Mengubah gambar menjadi RGB jika tidak
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # Menambahkan dimensi batch
    
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

# Halaman login
def login_page():
    st.title("Login untuk Mengakses Aplikasi")
    username = st.text_input("Username", "")
    password = st.text_input("Password", "", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.session_state.page = "Home"
        else:
            st.error("Username atau password salah!")

# Halaman Home
def home_page():
    st.title("SISTEM PAKAR PENDETEKSI PENYAKIT PNEUMONIA MENGGUNAKAN ALGORITMA CNN")
    st.subheader("Selamat Datang di Aplikasi Prediksi Pneumonia Berbasis AI 🩺")

    st.write("""
    *Aplikasi Prediksi Pneumonia* adalah alat bantu berbasis kecerdasan buatan yang dirancang untuk mendeteksi
    adanya penyakit pneumonia dari gambar X-ray paru-paru. Pneumonia adalah infeksi serius yang mengakibatkan
    peradangan pada kantong udara di paru-paru dan dapat berpotensi fatal jika tidak ditangani dengan benar.
    Aplikasi ini dibuat untuk membantu masyarakat umum maupun tenaga medis dalam melakukan deteksi dini
    dan memberikan gambaran cepat mengenai potensi pneumonia.
    """)
    
    st.markdown("---")

    st.subheader("Apa Itu Pneumonia?")
    image = Image.open('image/baner_pneumonia.jpg')
    st.image('image/baner_pneumonia.jpg', caption='',use_column_width=True)
    st.write("""
    Pneumonia adalah infeksi yang mempengaruhi kantong-kantong udara kecil di dalam paru-paru yang disebut alveoli.
    Pada kondisi ini, kantong-kantong udara tersebut dapat terisi oleh cairan atau nanah, sehingga menyebabkan
    batuk dengan lendir, demam, menggigil, dan kesulitan bernapas. Penyakit ini dapat disebabkan oleh berbagai jenis
    mikroorganisme, termasuk bakteri, virus, dan jamur.
    
    *Mengapa Penting untuk Deteksi Dini?*
    Deteksi dini sangat penting karena pneumonia dapat menyebabkan komplikasi serius terutama pada anak-anak, 
    orang tua, dan individu dengan sistem kekebalan tubuh yang lemah. Dengan deteksi dini melalui aplikasi ini, 
    diharapkan dapat membantu pengguna untuk segera mencari penanganan medis yang diperlukan.
    """)

    st.markdown("---")

    st.subheader("Cara Kerja Aplikasi")
    image = Image.open('image/tutor_tb.jpg')
    st.image('image/tutor_tb.jpg', caption='', use_column_width=True)
    st.write("""
    Aplikasi ini menggunakan model Convolutional Neural Network (CNN) yang dilatih khusus untuk membedakan
    antara gambar X-ray yang menunjukkan adanya pneumonia atau tidak. Berikut adalah cara kerja aplikasi ini:

    1. *Unggah Gambar X-ray Paru-paru*: Pengguna dapat mengunggah gambar X-ray yang ingin diprediksi.
    2. *Proses Prediksi*: Gambar yang diunggah akan diproses oleh model yang telah dilatih untuk mengenali pola-pola khas pneumonia.
    3. *Hasil Prediksi*: Aplikasi akan menampilkan hasil prediksi berupa kemungkinan adanya pneumonia dengan tingkat akurasi tertentu.
    4. *Saran*: Berdasarkan hasil prediksi dan tingkat akurasi, aplikasi akan memberikan rekomendasi tindakan yang dapat dilakukan.
    """)

    st.markdown("---")

    st.subheader("Penjelasan Tentang Akurasi")
    st.write("""
    Akurasi adalah ukuran seberapa benar atau sesuai hasil prediksi model dengan kondisi sebenarnya. 
    Dalam konteks aplikasi ini, akurasi membantu kita memahami seberapa yakin model dalam memprediksi 
    apakah gambar X-ray menunjukkan tanda-tanda pneumonia. Tingkat akurasi yang tinggi mengindikasikan 
    prediksi yang lebih dapat diandalkan.

    *Contoh Penafsiran Akurasi*:
    - Jika akurasi mendekati 100%, berarti model sangat yakin dengan hasil prediksinya.
    - Jika akurasi lebih rendah, hasil prediksi tetap bermanfaat namun disarankan untuk melakukan 
      pemeriksaan lanjutan atau konsultasi medis untuk memastikan diagnosis.
    """)
    st.markdown("---")
    
    st.subheader("Visualisasi Data")
    st.write("""
    Visualisasi Data disini menunjukan diagram guna memantau presentase ataupun konsistensi dari pelatihan model yang dilakukan.
    """)
    st.write("")
    image = Image.open('image/model_akurasi_pneumonia.jpg')
    st.image('image/model_akurasi_pneumonia.jpg', caption='Gambar Visualisasi Matrix Pelatihan Model', use_column_width=True)
    st.write("""
             Visualisasi ini menunjukkan kinerja model deep learning berdasarkan dataset pneumonia yang diambil dari Kaggle. Berikut adalah analisisnya:

1. *Model Accuracy (Grafik Kiri)*
- Grafik ini nunjukin gimana akurasi model selama pelatihan dan validasi.
- Awalnya, akurasi pelan-pelan naik, terus sekitar epoch ke-15 mulai stabil, dan di akhir akurasi pelatihan sama validasinya udah di atas 90%.
- Ini berarti model lumayan mantap, udah bisa ngeklasifikasi pneumonia dengan akurasi tinggi.
2. *Model Loss (Grafik Kanan)*
- Nah, grafik loss ini kebalikannya dari akurasi. Semakin kecil angkanya, semakin bagus.
- Di awal-awal pelatihan (epoch 0-10), loss turun tajam banget, terus makin lama makin stabil sampai akhir pelatihan.
- Menariknya, loss validasi malah sering lebih rendah dibanding loss pelatihan, jadi model tersebut nggak cuma jago di data latih, tapi juga keren di data yang belum pernah dilihat (validasi).
3. *Parameter yang Digunakan*
- Image Size    : Image Size yang dipakai ukuran 150x150 piksel. Ini udah cukup buat gambar X-ray biar fitur pentingnya keambil.
- Batch Size    : Jumlah Batch-nya 16, jadi setiap pelatihan ngolah 16 gambar sekaligus. Ini bikin model lebih fokus ke detail gambar tapi agak makan waktu.
- Epochs        : Pelatihan 40 kali (epoch), dan kelihatan di grafiknya model udah stabil sebelum 40, jadi udah cukup.
             """)
    st.markdown("---")
    image = Image.open('image/acc_perclass.png')
    st.image('image/acc_perclass.png', caption='Gambar Visualisasi Pelatihan berdasarkan tiap tiap kelas', use_column_width=True)
    st.write("""
    Tabel Akurasi Per Kelas
Apa isi tabel ini?
Tabel ini menunjukkan seberapa akurat model mengenali dua kategori, yaitu normal dan pneumonia:

- Normal: Akurasi model mencapai 94% (artinya dari semua gambar yang "normal," 94% diklasifikasi dengan benar).
- Pneumonia: Akurasi model lebih tinggi lagi, yaitu 98%.
Jumlah Sampel:
- Untuk tiap kategori, ada 203 gambar yang diuji.
    """)
    st.markdown("---")
    image = Image.open('image/confus_matrix.png')
    st.image('image/confus_matrix.png', caption='Gambar Visualisasi Confusion Matrix', use_column_width=True)
    st.write("""
    Confusion Matrix
1. Apa itu confusion matrix?
Ini adalah cara visual untuk melihat seberapa banyak prediksi model benar atau salah.

2. Penjelasan Matriks:

- Kotak biru tua: Prediksi benar (sesuai dengan label aslinya).
- Kotak biru muda: Prediksi salah.
Mari kita lihat angkanya:

- 190 (kiri atas): Gambar yang sebenarnya normal dan diprediksi normal.
- 13 (kanan atas): Gambar normal, tapi diprediksi salah sebagai pneumonia.
- 4 (kiri bawah): Gambar pneumonia, tapi diprediksi salah sebagai normal.
- 199 (kanan bawah): Gambar yang sebenarnya pneumonia dan diprediksi pneumonia. Apa arti hasil ini?

Model cukup bagus karena sebagian besar prediksi berada di kotak biru tua (prediksi benar).
Namun, ada 13 kesalahan untuk kategori normal dan 4 kesalahan untuk pneumonia.
    """)

import matplotlib.pyplot as plt

# Halaman Prediksi
def prediction_page():
    st.title("Halaman Prediksi Pneumonia")
    st.markdown("""    
    Unggah gambar X-ray paru-paru Anda untuk mendeteksi kemungkinan pneumonia.

    *Catatan:* Aplikasi ini bukan pengganti diagnosis medis, jangan jadikan hasil prediksi ini menjadi keputusan yang absolute.
 
    """)
    
    st.markdown("---")
    st.subheader("📋 Langkah-langkah:")
    st.write("""
    1. Klik tombol *'Browse Files'* di bawah ini untuk mengunggah gambar X-ray paru-paru Anda.
    2. Tunggu beberapa saat hingga aplikasi selesai memproses.
    3. Lihat hasil prediksi di bawah beserta tingkat kepercayaan model.
    """)
    
    st.subheader("📤 Unggah Gambar X-ray")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    if model is not None:
        predicted_class, confidence = predict(image, model)
        st.write(f"Prediksi: {class_names[predicted_class].strip()}")  # Menghilangkan nomor kelas
        st.write(f"Tingkat Kepercayaan: {confidence:.2f}%")
    else:
        st.error("Model tidak tersedia. Pastikan model dimuat dengan benar.")

# Halaman Tentang
def about_page():
    st.title("Tentang Pengembang")
    
    # Deskripsi Pengembang
    st.write("""
    **Perkenalkan saya, Fajar Pangestu Amandaru.**  
    Saya adalah mahasiswa Teknik Informatika di Universitas Indraprasta PGRI.  
    Aplikasi ini dibuat sebagai bagian dari tugas akhir saya, yang bertujuan untuk 
    mendeteksi penyakit pneumonia dengan teknologi kecerdasan buatan.
    """)

    # Kolom untuk media sosial
    st.subheader("Social Media")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📷 Instagram")
        st.markdown("[fjrpangestu](https://www.instagram.com/fjrpangestu)")

    with col2:
        st.markdown("#### 🔗 LinkedIn")
        st.markdown("[Fajar Pangestu Amandaru](https://www.linkedin.com/in/fajarpangestuamandaru/)")

    with col3:
        st.markdown("#### 📧 Email")
        st.markdown("[fajar.faps@gmail.com](mailto:fajar.faps@gmail.com)")

    # Tambahkan gambar jika diinginkan (opsional)
    
    image = Image.open('image/foto_profil.jpg')
    st.image('image/foto_profil.jpg', caption='Fajar Pangestu Amandaru', use_column_width=True)

    # Tambahkan gaya visual lainnya
    st.markdown("---")
    st.markdown(
        """
        <style>
        h1, h2, h3, h4 {
            color: #FF4B4B;
        }
        .stMarkdown p {
            line-height: 1.8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
# Sidebar

def sidebar():
    st.sidebar.title("Navigasi")

    if st.sidebar.button("Home"):
        st.session_state.page = "Home"
    if st.sidebar.button("Prediksi"):
        st.session_state.page = "Prediksi"
    if st.sidebar.button("About"):
        st.session_state.page = "About"

# Menentukan tampilan berdasarkan halaman yang dipilih
def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'page' not in st.session_state:
        st.session_state.page = 'Login'

    if st.session_state.logged_in:
        sidebar()

        if st.session_state.page == "Home":
            home_page()
        elif st.session_state.page == "Prediksi":
            prediction_page()
        elif st.session_state.page == "About":
            about_page()
        else:
            st.warning("Halaman tidak dikenal.")
    else:
        login_page()

if __name__ == "__main__":
    main()
