import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import streamlit as st
import cv2
import random


# --- โหลดและเตรียมภาพสำหรับทุกโมเดล ---
def load_and_prep_image(uploaded_file, target_size=(32,32), channels=1):
    """
    uploaded_file : ไฟล์รูปภาพจาก uploader
    target_size   : ขนาดที่ต้องการ (width, height)
    channels      : 1 = grayscale, 3 = RGB
    """
    image = Image.open(uploaded_file).convert("L").resize(target_size)
    img_array = np.array(image).astype('float32') / 255.0

    # เพิ่ม channels
    if channels == 3:
        img_array = np.stack([img_array]*3, axis=-1)
    else:
        img_array = np.expand_dims(img_array, -1)

    # เพิ่ม batch dimension
    img_array = np.expand_dims(img_array, 0)
    return img_array

# --- ทำนายและแสดงภาพแบบสุ่ม ---
def predict_and_plot_random(model, x_test, y_test, class_names, target_size=(28,28), channels=1):
    """
    x_test : ชุดทดสอบ
    channels : จำนวน channels ของโมเดล
    target_size : ขนาดภาพ input ของโมเดล
    """
    idx = random.randint(0, len(x_test)-1)
    img = x_test[idx]
    true_label = y_test[idx]

    # Resize และจัด channels
    img_resized = cv2.resize(img, target_size)
    if channels == 3:
        img_input = np.stack([img_resized]*3, axis=-1)
    else:
        img_input = np.expand_dims(img_resized, -1)
    img_input = img_input.astype('float32') / 255.0
    img_input = np.expand_dims(img_input, 0)

    # Predict
    predictions = model.predict(img_input)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    acc = predictions[0][predicted_class_index]*100
    true_name = class_names[true_label]

    # แสดงภาพ
    st.image(img_resized, caption=f"Predict: {predicted_class_name} | True: {true_name} | Probability: {acc:.2f}%", width=200)

    # แสดง bar plot
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=class_names, y=predictions[0], palette="viridis")
    plt.xticks(rotation=45)
    plt.ylim(0, 1.05)
    plt.ylabel("Probability")
    plt.title("Prediction Confidence")
    st.pyplot(fig)

# --- แสดง bar plot ของ prediction ---
def plot_prediction_bar(predictions, class_names):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=class_names, y=predictions[0], palette="viridis")
    plt.xticks(rotation=45)
    plt.ylim(0, 1.05)
    plt.ylabel("Probability")
    plt.title("Prediction Confidence")
    st.pyplot(fig)
