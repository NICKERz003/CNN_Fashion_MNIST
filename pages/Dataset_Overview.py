import streamlit as st
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np

st.title("📊 Fashion MNIST Dataset Overview")

# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

st.subheader("🩵 ข้อมูลเบื้องต้น")
st.markdown(f"""
- จำนวนภาพฝึก (train): **{x_train.shape[0]}**
- จำนวนภาพทดสอบ (test): **{x_test.shape[0]}**
- ขนาดภาพ: **{x_train.shape[1]}×{x_train.shape[2]}**
- จำนวนคลาส: **{len(class_names)}**
""")

# Show random samples
st.subheader("🖼️ ตัวอย่างข้อมูลจาก Dataset")
cols = st.columns(10)
for i, col in enumerate(cols):
    idx = np.random.randint(0, len(x_train))
    col.image(x_train[idx], caption=class_names[y_train[idx]], width=64)

# Data preprocessing explanation
st.subheader("⚙️ ขั้นตอนการเตรียมข้อมูล (Data Preparation)")
st.markdown("""
- Normalize ค่าพิกเซลให้อยู่ระหว่าง 0-1  
- แปลง label เป็น one-hot vector  
- สำหรับโมเดล pretrained เช่น VGG16 / ResNet50 / MobileNetV2  
  จะมีการ **resize และ replicate grayscale เป็น 3 channels**
""")
