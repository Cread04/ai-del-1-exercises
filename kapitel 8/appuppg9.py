import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions


st.title("My Image Classifier AI")
st.write("upload a picture and I will try to tell you what it is!")

# Load the model
# we are using cache_resource to avoid reloading the model on every interaction
@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

model = load_model()

# file uploader
file = st.file_uploader("Välj en bild (jpg eller png)", type=["jpg", "png", "jpeg"])

if file is not None:
    # show the uploaded image
    image = Image.open(file)
    st.image(image, caption='Din bild', use_column_width=True)
    
    st.write("Analyserar...")

    # prepare the image for the model
    # MobileNetV2 wants 224x224 images
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # convert the image to a numpy array
    img_array = np.asarray(image)
    # expand the dimensions to (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # use mobilenet preprocessing
    processed_image = preprocess_input(img_array.astype(np.float32))

    # predict with the model
    predictions = model.predict(processed_image)
    decoded = decode_predictions(predictions, top=3)[0]

    # show the results
    st.success(f"Jag tror det är en: **{decoded[0][1].replace('_', ' ')}**")
    st.write(f"Säkerhet: {decoded[0][2]*100:.1f}%")
    
    # Show top 3 predictions
    st.write("---")
    st.write("**Andra gissningar:**")
    for i, (id, label, prob) in enumerate(decoded[1:]):
        st.write(f"{i+2}. {label}: {prob*100:.1f}%")