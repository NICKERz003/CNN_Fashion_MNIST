import streamlit as st
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model
from PIL import Image
from utils.helpers import load_and_prep_image, predict_and_plot_random, plot_prediction_bar

st.title("MobileNetV2")

# --- โหลด dataset ---
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_names = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]

# Normalize สำหรับแสดงผล
x_test_display = x_test / 255.0
x_test_display = np.expand_dims(x_test_display, axis=-1)

# --- โหลดโมเดล ---
model = load_model("models/fashion_mobilenetv2.h5")

# --- ทำนายภาพแบบสุ่ม ---
st.subheader("ทำนายภาพแบบสุ่มจากชุดทดสอบ")
if st.button("สุ่มภาพ"):
    predict_and_plot_random(model, x_test, y_test, class_names, target_size=(96,96), channels=3)

# --- อัปโหลดภาพเพื่อทำนาย ---
st.subheader("อัพโหลดภาพเพื่อทำนาย")
uploaded = st.file_uploader("อัพโหลดรูปภาพ (.jpg, .png, .jpeg)", type=["jpg","png","jpeg"])
if uploaded:
    # โหลดและเตรียมภาพให้ตรงกับ input ของ MobileNetV2 (32x32, 3 channels)
    img_arr = load_and_prep_image(uploaded, target_size=(96,96), channels=3)
    image = Image.open(uploaded).convert("L").resize((96,96))
    st.image(image, caption='รูปภาพที่ปรับปรุงแล้ว (96x96 3 channels)', width=150)

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
st.image("assets/MobileNetV2_accuracy_loss.png")
st.subheader("Confusion Matrix")
st.image("assets/MobileNetV2_confusion_matrix.png")
st.subheader("Classification Report")
st.image("assets/MobileNetV2_classification_report.png")
st.subheader("Precision, Recall และ F1-Score")
st.image("assets/MobileNetV2_Precision_Recall_F1.png")
# --- ตัวอย่างโค้ดเทรนโมเดล ---
st.subheader("💻 ตัวอย่างโค้ดเทรนโมเดล")
st.code("""
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

# Load data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Resize to 96x96 for MobileNetV2 input
x_train = np.stack([x_train]*3, axis=-1)
x_test = np.stack([x_test]*3, axis=-1)
x_train = np.array([cv2.resize(img, (96,96)) for img in x_train]).astype('float32') / 255.0
x_test = np.array([cv2.resize(img, (96,96)) for img in x_test]).astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Load pretrained MobileNetV2
base = MobileNetV2(include_top=False, input_shape=(96,96,3), weights='imagenet', pooling='avg')

# Freeze base layers
base.trainable = False

# Add custom classification head
x = Dropout(0.3)(base.output)
output = Dense(10, activation='softmax')(x)
model = Model(inputs=base.input, outputs=output)

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True)
model.fit(datagen.flow(x_train, y_train, batch_size=64),
          validation_data=(x_test, y_test),
          epochs=10)
""", language="python")
