import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model
from utils.helpers import load_and_prep_image, predict_and_plot_random, plot_prediction_bar
from PIL import Image

st.title("Simple CNN Model")
# โหลด dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_names = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]

# Normalize
x_test = x_test / 255.0
x_test = np.expand_dims(x_test, axis=-1)

# โหลดโมเดล
model = load_model("models/fashion_mnist_cnn_model_20E.h5")

# 🔹 ทำนายภาพแบบสุ่ม
st.subheader("ทำนายภาพแบบสุ่มจากชุดทดสอบ")
if st.button("สุ่มภาพ"):
    predict_and_plot_random(model, x_test, y_test, class_names)

# 🔹 อัปโหลดภาพเพื่อทำนาย
st.subheader("อัพโหลดภาพเพื่อทำนาย")
uploaded = st.file_uploader("อัพโหลดรูปภาพ (.jpg, .png, .jpeg)", type=["jpg","png","jpeg"])
if uploaded:
    # โหลดและเตรียมภาพ
    img_arr = load_and_prep_image(uploaded, target_size=(28,28))
    image = Image.open(uploaded).convert("L").resize((28,28))
    st.image(image, caption='รูปภาพที่ปรับปรุงแล้ว (28x28 Grayscale)', width=150)

    # ทำนายผล
    predictions = model.predict(img_arr)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]

    # แสดงผลลัพธ์
    st.success(f"**การทำนายผล:** {predicted_class_name}")
    st.write(f"ความน่าจะเป็นสูงสุด: **{predictions[0][predicted_class_index]*100:.2f}%**")

    # แสดง bar plot
    plot_prediction_bar(predictions, class_names)

# 🔹 กราฟหลังเทรนโมเดล
st.subheader("กราฟหลังเทรนโมเดล")
st.image("assets/cnn_accuracy_loss.png")
st.subheader("Confusion Matrix")
st.image("assets/cnn_confusion_matrix.png")
st.subheader("Classification Report")
st.image("assets/cnn_classification_report.png")

# 🔹 ตัวอย่างโค้ดเทรนโมเดล
st.subheader("💻 ตัวอย่างโค้ดเทรนโมเดล")
st.code("""
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
        
# โหลดข้อมูล
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# กำหนดชื่อคลาส (Labels)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

# เดิม x_train.shape: (60000, 28, 28)
# เปลี่ยนเป็น: (60000, 28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        
from tensorflow.keras.utils import to_categorical
# แปลง 5 เป็น [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
        
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
        
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
        
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=32 ,validation_data=(x_test, y_test))
""", language="python")


