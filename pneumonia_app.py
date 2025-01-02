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
    adanya penyakit pneumonia dari gambar X-ray paru-paru. Aplikasi ini membantu deteksi dini penyakit.
    """)

    st.markdown("---")

    st.subheader("Apa Itu Pneumonia?")
    st.image('image/baner_pneumonia.jpg', caption='', use_column_width=True)

    st.write("""
    Pneumonia adalah infeksi paru-paru yang dapat mengancam nyawa jika tidak ditangani dengan tepat.
    Deteksi dini sangat penting untuk mencegah komplikasi serius.
    """)

    st.markdown("---")

    st.subheader("Cara Kerja Aplikasi")
    st.image('image/tutor_tb.jpg', caption='', use_column_width=True)
    st.write("""
    1. Unggah gambar X-ray.
    2. Model memproses gambar untuk memprediksi keberadaan pneumonia.
    3. Hasil prediksi ditampilkan dengan tingkat kepercayaan.
    """)

# Halaman Prediksi
def prediction_page():
    st.title("🔍 Prediksi Pneumonia")

    uploaded_file = st.file_uploader("Unggah gambar X-ray paru-paru", type=["jpg", "jpeg", "png"])

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
    st.title("Tentang Aplikasi")
    st.write("""
    Aplikasi ini dikembangkan untuk membantu mendeteksi penyakit pneumonia dari gambar X-ray paru-paru.
    Sistem ini menggunakan model deep learning untuk membedakan gambar normal dan yang menunjukkan tanda pneumonia.
    """)

    st.markdown("---")

    st.subheader("Catatan Penting")
    st.write("""
    - Aplikasi ini hanya memberikan prediksi awal.
    - Hasil aplikasi perlu dikonfirmasi oleh tenaga medis.
    """)

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
