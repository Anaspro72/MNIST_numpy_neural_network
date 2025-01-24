import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import pickle


with open("mnist_model.pkl", "rb") as file:
    model_parameters = pickle.load(file)

W1, b1 = model_parameters["W1"], model_parameters["b1"]
W2, b2 = model_parameters["W2"], model_parameters["b2"]
W3, b3 = model_parameters["W3"], model_parameters["b3"]


def preprocess_image(image):
    image = image.convert("L")  
    image = image.resize((28, 28))  
    image_array = np.array(image) / 255.0  
    return image_array.flatten()[:, None]  


def predict_mnist(image_array, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(W1, image_array) + b1
    A1 = np.maximum(0, Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = np.maximum(0, Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = np.exp(Z3) / np.sum(np.exp(Z3), axis=0)
    return np.argmax(A3)


st.set_page_config(page_title="MNIST Digit Classifier", page_icon="✏️", layout="centered")

st.title("🖌️ MNIST Digit Classifier")
st.subheader("Upload a handwritten digit image or enter a URL, and let the model predict the digit!")
st.write("This app uses a trained neural network built from scratch with NumPy. 🚀")


if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None


st.divider()  
upload_option = st.radio("🎛️ Choose input method:", ("Upload Image", "Enter Image URL"))

if upload_option == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload a digit image (e.g., 28x28 pixels):", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        st.session_state.uploaded_image = Image.open(uploaded_file)
elif upload_option == "Enter Image URL":
    image_url = st.text_input("🌐 Enter the image URL:")
    if image_url:
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            st.session_state.uploaded_image = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error(f"Error fetching the image: {e}")
            st.session_state.uploaded_image = None


if st.button("🔄 Clear"):
    st.session_state.uploaded_image = None


if st.session_state.uploaded_image:
    st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_column_width=True)
    st.write("🖼️ Processing the image...")

    
    preprocessed_image = preprocess_image(st.session_state.uploaded_image)

    
    prediction = predict_mnist(preprocessed_image, W1, b1, W2, b2, W3, b3)
    st.success(f"🎉 Prediction: The digit is **{prediction}**!")
else:
    st.info("🤔 Upload an image or provide a URL to start!")

st.divider()  
st.write("📝 **Note:** This app is built with Streamlit and demonstrates deploying a NumPy-based neural network. 😊")
