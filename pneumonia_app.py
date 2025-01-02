import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model_path = 'keras_model_pneumonia.h5'  # Ganti dengan path model yang sesuai
class_names_path = 'labels_pneumonia.txt'  # Ganti dengan path file label yang sesuai

# Memuat model dan label
try:
    model = load_model(model_path)
    class_names = [line.strip() for line in open(class_names_path, "r")]
except Exception as e:
    st.error(f"Error loading model or labels: {e}")

# Tentukan ukuran gambar
IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224

# Fungsi untuk memprediksi gambar
def predict(image, model):
    try:
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

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
    """)
    st.markdown("---")
    st.subheader("Apa Itu Pneumonia?")
    st.image('image/baner_pneumonia.jpg', use_column_width=True)
    st.write("""    
    Pneumonia adalah infeksi yang mempengaruhi kantong-kantong udara kecil di dalam paru-paru yang disebut alveoli.
    """)
    st.markdown("---")
    st.subheader("Cara Kerja Aplikasi")
    st.image('image/tutor_tb.jpg', use_column_width=True)
    st.write("""    
    Aplikasi ini menggunakan model Convolutional Neural Network (CNN) yang dilatih khusus untuk membedakan
    antara gambar X-ray yang menunjukkan adanya pneumonia atau tidak.
    """)
    st.markdown("---")
    st.subheader("Visualisasi Data")
    st.image('image/model_akurasi_pneumonia.jpg', use_column_width=True)

# Halaman Prediksi
def prediction_page():
    st.title("🔍 Prediksi Gambar X-ray Pneumonia")
    uploaded_file = st.file_uploader("Unggah gambar X-ray", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        predicted_class, confidence = predict(image, model)
        if predicted_class is not None:
            st.markdown(f"**Prediksi Kelas:** {class_names[predicted_class]}")
            st.markdown(f"**Akurasi:** {confidence:.2f}%")

# Halaman Tentang
def about_page():
    st.title("Tentang Aplikasi")
    st.write("""
    Aplikasi ini dikembangkan untuk membantu mendeteksi penyakit pneumonia dari gambar X-ray paru-paru.
    """)
    st.markdown("---")
    st.subheader("Kriteria Aplikasi")
    st.write("""    
    - Aplikasi ini hanya berfungsi untuk memberikan informasi sementara mengenai kemungkinan adanya pneumonia.
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
        sidebar()
        if st.session_state.page == "Home":
            home_page()
        elif st.session_state.page == "Prediksi":
            prediction_page()
        elif st.session_state.page == "About":
            about_page()
    else:
        login_page()

if __name__ == "__main__":
    main()
