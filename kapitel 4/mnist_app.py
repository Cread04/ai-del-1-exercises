import streamlit as st
import numpy as np
import cv2  # OpenCV för bildhantering
from PIL import Image, ImageOps
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

#  loading and training the model can be time-consuming, so we cache it
@st.cache_resource
def load_and_train_model():
    st.write(" Laddar MNIST-data och tränar modellen")
    
    # Fetch MNIST data
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    
    # Normalize (0-255 -> 0-1)
    X = X / 255.0
    
    X_full = X 
    y_full = y
  
    # SVM is good for this task, but training on full MNIST can be slow
    model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    model.fit(X_full, y_full)
    
    st.success("Modellen har tränats")
    return model

#PREPROCESSING 
def process_image(image):
    """
    Konverterar ett mobilfoto till MNIST-format:
    28x28 pixlar, gråskala, svart bakgrund, vit siffra, centrerad.
    """
    # convert to grayscale
    img = image.convert('L')
    
   
    # invert the colors
    # we check if the image is light (white paper). If so, we invert it.
    img_array = np.array(img)
    if img_array.mean() > 127:  
        img = ImageOps.invert(img)
    

    img_np = np.array(img)
    kernel = np.ones((3,3), np.uint8) 
   
    img_np = cv2.dilate(img_np, kernel, iterations=1)
    
  
    img = Image.fromarray(img_np)
  

    #find the number's bounding box to center it
    # we are using OpenCV to find contours
    img_np = np.array(img)
    _, thresh = cv2.threshold(img_np, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_img = Image.new('L', (28, 28), 0)
    
    if contours:
        #find the largest contour which should be the digit
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        # crop the image to the bounding box
        digit = img.crop((x, y, x+w, y+h))
        
        # Resize the digit to fit in a 20x20 box while maintaining aspect ratio
        max_size = 20
        ratio = max_size / max(digit.size)
        new_size = (int(digit.size[0] * ratio), int(digit.size[1] * ratio))
        
        # Resize using LANCZOS filter for better quality
        digit = digit.resize(new_size, Image.Resampling.LANCZOS)
        
        # center the digit in the 28x28 image
        bg_w, bg_h = final_img.size
        img_w, img_h = digit.size
        offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
        final_img.paste(digit, offset)
        
    else:
        # Fallback if no number is found: resize the original image
        final_img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Normalize pixel values to 0-1 and reshape for model input
    final_array = np.array(final_img).reshape(1, -1) / 255.0
    
    return final_array, final_img

# Streamlit app
st.title("Handskrivna Siffror och Mnist-Modell")
st.write("""
Ladda upp en bild på en handskriven siffra (0-9). 
Tips: Skriv med en tjock svart penna på vitt papper för bäst resultat.
""")

# Load the model
model = load_and_train_model()

# upload the file
file = st.file_uploader("Ladda upp bild", type=["jpg", "png", "jpeg"])

if file is not None:
    # show the original image
    image = Image.open(file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Din bild", use_column_width=True)
    
    # Process the image
    processed_data, processed_image = process_image(image)
    
    with col2:
        # Show the processed image 
        st.image(processed_image, caption="Vad modellen ser (28x28)", width=150)
    
    # Button to predict
    if st.button("Prediktera Siffra"):
        prediction = model.predict(processed_data)[0]
        probs = model.predict_proba(processed_data)
        confidence = probs.max()
        
        st.markdown(f"## Jag tror det är en: **{prediction}**")
        st.write(f"Säkerhet: {confidence:.1%}")
        
        # Show probability distribution
        st.bar_chart(pd.DataFrame(probs.T, columns=["Sannolikhet"], index=list("0123456789")))

# streamlit run mnist_app.py
