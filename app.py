import streamlit as st
import numpy as np
import cv2
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import preprocess_input
import tensorflow as tf
import requests
from PIL import Image
import io
import matplotlib.pyplot as plt

# Load models and word index
feature_extraction_model = tf.keras.models.load_model('feature_extraction_model.h5')
captioning_model = tf.keras.models.load_model('mymodel.keras')
with open('word_index.pkl', 'rb') as f:
    word_index = pickle.load(f)

max_length = 44  # Replace with your actual max_length

def idx_to_word(integer, word_index):
    for word, index in word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, word_index, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [word_index.get(word, 0) for word in in_text.split()]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, word_index)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break

    in_text = in_text.replace("startseq", "").replace("endseq", "")
    return in_text

def extract_features(image, model, target_size=(299, 299)):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

def generate_captions(image):
    feature = extract_features(image, feature_extraction_model)
    captions = predict_caption(captioning_model, feature, word_index, max_length)
    return image, captions

# Streamlit app
st.title('Image Caption Generator')

option = st.selectbox("Choose how to provide the image:", ("Upload Image", "Image URL"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            # Check for valid image mode
            if image.mode not in ["RGB", "L"]:
                st.error("Unsupported image mode. Please upload an RGB or grayscale image.")
            else:
                image, captions = generate_captions(image)
                # Display results
                st.image(image, caption='Uploaded Image', use_column_width=True)
                st.write("### Predicted Captions")
                st.write(captions)
        except Exception as e:
            st.error(f"Error processing image: {e}")

elif option == "Image URL":
    image_url = st.text_input("Enter the URL of an image:")

    if image_url:
        try:
            response = requests.get(image_url)
            response.raise_for_status()  # Check if request was successful
            image = Image.open(io.BytesIO(response.content))
            # Check for valid image mode
            if image.mode not in ["RGB", "L"]:
                st.error("Unsupported image mode. Please provide an RGB or grayscale image URL.")
            else:
                # Generate captions
                image, captions = generate_captions(image)
                # Display results
                st.image(image, caption='Image from URL', use_column_width=True)
                st.write("### Predicted Caption")
                st.write(captions)
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching image from URL: {e}")
        except IOError:
            st.error("Error processing image from URL. Ensure the URL points to a valid image file.")
