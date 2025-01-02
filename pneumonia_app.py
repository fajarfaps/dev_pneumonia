import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model_path = 'keras_model.h5'  # Ganti dengan path model yang sesuai
model = load_model(model_path)

# Tentukan ukuran gambar dan nama kelas
IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224
class_names = open("labels.txt", "r").readlines()
# Fungsi untuk memprediksi gambar
def predict(image, model):
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
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
            # Set session state to logged in and redirect to home page
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
    image = Image.open("image/baner_pneumonia.jpg")
    st.image('image/baner_pneumonia.jpg', caption='', use_column_width=True)

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
    image = Image.open("image/tutor_tb.jpg")
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
    """)

    st.markdown("---")
    
    st.subheader("Visualisasi Data")
    st.write("""    
    Visualisasi Data disini menunjukan diagram guna memantau presentase ataupun konsistensi dari pelatihan model yang dilakukan.
    """)
    st.write("")
    image = Image.open("image/model_akurasi_pneumonia.jpg")
    st.image('image/model_akurasi_pneumonia.jpg', caption='Gambar Visualisasi Matrix Pelatihan Model', use_column_width=True)

# Halaman Prediksi
def prediction_page():
    st.markdown("<h1 class='black-text'>🔍 Prediksi Gambar Acne</h1>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='black-text'>🧠 Deskripsi Model</h2>", unsafe_allow_html=True)
    st.markdown("""
        <p class='black-text'>
        Prediksi dilakukan menggunakan model deep learning yang telah dilatih menggunakan data gambar wajah 
        dengan dan tanpa acne. Model ini dapat memprediksi dengan tingkat akurasi yang tinggi berdasarkan fitur-fitur 
        yang dipelajari selama pelatihan.
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 class='black-text'>🖼️ Input</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Unggah gambar acne", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)
        
        predicted_class, confidence = predict(image, model)
        
        class_description = {
            0: "Jerawat Level 1: Jerawat kecil yang terlihat di beberapa area kulit. Biasanya tidak meradang.",
            1: "Jerawat Level 2: Jerawat yang lebih besar dan lebih banyak, mungkin sedikit meradang.",
            2: "Jerawat Level 3: Jerawat yang parah dengan peradangan yang signifikan, memerlukan perawatan medis.",
            3: "Tidak Berjerawat: Kulit wajah yang bersih tanpa adanya jerawat atau komedo."
        }
        
        st.markdown(f"<p class='black-text'><strong>Prediksi Kelas:</strong> {class_names[predicted_class]}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='black-text'><strong>Akurasi:</strong> {confidence:.2f}%</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='black-text'><strong>Deskripsi:</strong> {class_description[predicted_class]}</p>", unsafe_allow_html=True)


# Halaman Tentang
def about_page():
    st.title("Tentang Aplikasi")
    st.write("""
    Aplikasi ini dikembangkan untuk membantu mendeteksi penyakit pneumonia dari gambar X-ray paru-paru.
    Sistem ini menggunakan model deep learning yang dilatih dengan dataset X-ray paru-paru untuk membedakan antara gambar normal dan yang menunjukkan tanda-tanda pneumonia.
    """)
    
    st.markdown("---")

    st.subheader("Kriteria Aplikasi")
    st.write("""    
    - Aplikasi ini hanya berfungsi untuk memberikan informasi sementara mengenai kemungkinan adanya pneumonia.
    - Hasil dari aplikasi ini perlu dikonfirmasi lebih lanjut oleh tenaga medis.
    """)

# Sidebar kustom
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
        # Menampilkan sidebar kustom
        sidebar()

        # Menampilkan halaman sesuai menu
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
