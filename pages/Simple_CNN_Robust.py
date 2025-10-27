import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model
from utils.helpers import load_and_prep_image, predict_and_plot_random, plot_prediction_bar
from PIL import Image

st.title("Simple CNN Robust , เพิ่ม BatchNormalization Dropout และ Data Augmentation")
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
model = load_model("models/fashion_mnist_cnn_model_robust.h5")

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
st.image("assets/cnn_robust_accuracy_loss.png")
st.subheader("Confusion Matrix")
st.image("assets/cnn_robust_confusion_matrix.png")
st.subheader("Classification Report")
st.image("assets/cnn_robust_classification_report.png")

# 🔹 ตัวอย่างโค้ดเทรนโมเดล
st.subheader("💻 ตัวอย่างโค้ดเทรนโมเดล")
st.code("""
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator # เพิ่มไลบรารีนี้

#1.เตรียมข้อมูล
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


#2.ปรับปรุงโครงสร้างโมเดล (เพิ่ม BN และ Dropout)
model = Sequential([
    # Convolutional Block 1
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(), # เพิ่ม BatchNormalization เพื่อความเสถียร
    MaxPooling2D((2, 2)),

    # Convolutional Block 2
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Classification Block
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5), # เพิ่ม Dropout 50% เพื่อลด Overfitting
    Dense(10, activation='softmax')
])
model.summary()

# Compile Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#3.เทรนด้วย Data Augmentation

# กำหนด Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,        # หมุนภาพ 10 องศา
    zoom_range=0.3,           # ซูมภาพ 30%
    width_shift_range=0.1,    # เลื่อนภาพตามแนวกว้าง 10%
    height_shift_range=0.1,   # เลื่อนภาพตามแนวสูง 10%
    horizontal_flip=False     # ไม่พลิกแนวนอนสำหรับเสื้อผ้า
)

# เตรียม data generator
datagen.fit(x_train)

# Train Model
# ใช้ .flow() เพื่อเทรนด้วยข้อมูล Augmentation
history = model.fit(datagen.flow(x_train, y_train, batch_size=64), # Batch Size 64 เหมาะกับ Augmentation
                    epochs=20,
                    validation_data=(x_test, y_test),
                    steps_per_epoch=len(x_train) // 64) # ปรับ steps_per_epoch ตาม Batch Size
""", language="python")