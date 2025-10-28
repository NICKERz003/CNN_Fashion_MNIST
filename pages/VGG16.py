import streamlit as st
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
# เพิ่ม path ของโฟลเดอร์ utils ให้ Python หาเจอ
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from helpers import load_and_prep_image, predict_and_plot_random, plot_predictio

st.title("VGG16")

# --- โหลด dataset ---
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_names = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]

# Normalize สำหรับแค่ display (VGG16 ใช้ฟังก์ชัน preprocessing ของ helper)
x_test_display = x_test / 255.0
x_test_display = np.expand_dims(x_test_display, axis=-1)

# --- สร้างโมเดล architecture ใหม่เหมือนตอนเทรน ---
TARGET_SIZE = (32, 32)
NUM_CLASSES = 10

vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
for layer in vgg_base.layers:
    layer.trainable = False

model = Sequential([
    vgg_base,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# --- โหลด weights ที่เราเทรนมา ---
model.load_weights("models/fashion_mnist_vgg16_transfer.weights.h5")
# --- ทำนายภาพแบบสุ่ม ---
st.subheader("ทำนายภาพแบบสุ่มจากชุดทดสอบ")
if st.button("สุ่มภาพ"):
    predict_and_plot_random(model, x_test, y_test, class_names, target_size=(32,32), channels=3)

# --- อัปโหลดภาพเพื่อทำนาย ---
st.subheader("อัพโหลดภาพเพื่อทำนาย")
uploaded = st.file_uploader("อัพโหลดรูปภาพ (.jpg, .png, .jpeg)", type=["jpg","png","jpeg"])
if uploaded:
    # โหลดและเตรียมภาพให้ตรงกับ input ของ VGG16 (32x32, 3 channels)
    img_arr = load_and_prep_image(uploaded, target_size=(32,32), channels=3)
    image = Image.open(uploaded).convert("RGB").resize((32,32))
    st.image(image, caption='รูปภาพที่ปรับปรุงแล้ว (32x32 3 channels)', width=150)

    # --- ทำนายผล ---
    predictions = model.predict(img_arr)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]

    # แสดงข้อความผลลัพธ์
    st.success(f"**การทำนายผล:** {predicted_class_name}")
    st.write(f"ความน่าจะเป็นสูงสุด: **{predictions[0][predicted_class_index]*100:.2f}%**")

    # --- แสดง bar plot ของ prediction confidence ---
    plot_prediction_bar(predictions, class_names)

# --- กราฟหลังเทรนโมเดล ---
st.subheader("กราฟหลังเทรนโมเดล")
st.image("assets/vgg16_accuracy_loss.png")
st.subheader("Confusion Matrix")
st.image("assets/vgg16_confusion_matrix.png")
st.subheader("Classification Report")
st.image("assets/vgg16_classification_report.png")

# --- ตัวอย่างโค้ดเทรนโมเดล ---
st.subheader("💻 ตัวอย่างโค้ดเทรนโมเดล")
st.code("""
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
import cv2

# โหลดข้อมูล Fashion MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

TARGET_SIZE = (32, 32)
NUM_CLASSES = 10

# ฟังก์ชัน Preprocessing
def preprocess_vgg16_cv2(images):
    resized_images = np.array([cv2.resize(img, TARGET_SIZE) for img in images])
    if resized_images.ndim == 3:
        resized_images = np.expand_dims(resized_images, axis=-1)
    images_3channel = np.repeat(resized_images, 3, axis=-1)
    images_normalized = images_3channel.astype('float32') / 255.0
    return images_normalized

x_train_vgg = preprocess_vgg16_cv2(x_train)
x_test_vgg = preprocess_vgg16_cv2(x_test)
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

# สร้างโมเดล VGG16 + Classification Head
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
for layer in vgg_base.layers:
    layer.trainable = False

model = Sequential([
    vgg_base,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train_vgg, y_train, epochs=30, batch_size=32, validation_data=(x_test_vgg, y_test))
""", language="python")

