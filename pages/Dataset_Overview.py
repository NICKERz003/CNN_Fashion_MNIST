import streamlit as st
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np

st.title("Fashion MNIST Dataset Overview")

# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


st.markdown("""
Fashion MNIST เป็นชุดข้อมูลรูปภาพที่พัฒนาโดย **Zalando Research** เพื่อเป็น **ชุดข้อมูลทดแทนที่ซับซ้อนกว่า** ชุดข้อมูล MNIST ดั้งเดิม (ที่ใช้ภาพตัวเลขลายมือ) 

ชุดข้อมูลนี้ถูกใช้กันอย่างแพร่หลายสำหรับเป็นด่านแรกในการทดสอบและเปรียบเทียบประสิทธิภาพของอัลกอริทึม **Machine Learning** และ **Deep Learning** โดยเฉพาะอย่างยิ่ง **Convolutional Neural Networks (CNN)**
""")

st.image("https://github.com/zalandoresearch/fashion-mnist/raw/master/doc/img/fashion-mnist-sprite.png")

st.subheader("ลักษณะและโครงสร้างของข้อมูล")
st.markdown(f"""
Dataset นี้ประกอบด้วยภาพถ่ายเสื้อผ้าและเครื่องประดับ **10 ประเภท** โดยมีลักษณะสำคัญดังนี้:
* **ประเภทภาพ:** เป็นภาพ **ขาวดำ (Grayscale)** มีเพียง 1 Channel สี
* **ความละเอียดต่ำ:** แต่ละภาพมีขนาดเพียง **{x_train.shape[1]} × {x_train.shape[2]} พิกเซล**
* **ความท้าทาย:** ภาพมีขนาดเล็ก แต่ความหลากหลายและรายละเอียดของเสื้อผ้า (เช่น ความแตกต่างระหว่าง 'Pullover' และ 'Shirt') นั้นซับซ้อนกว่าการแยกแยะตัวเลขมาก ทำให้โมเดลต้องเรียนรู้คุณลักษณะระดับสูงขึ้นเพื่อจำแนก
""")

st.subheader("ข้อมูลเบื้องต้น")
st.markdown(f"""
- **จำนวนภาพรวมทั้งหมด:** **70,000** ภาพ
- **จำนวนภาพฝึก (train):** **{x_train.shape[0]}** ภาพ
- **จำนวนภาพทดสอบ (test):** **{x_test.shape[0]}** ภาพ
- **ขนาดภาพ:** **{x_train.shape[1]}×{x_train.shape[2]}** พิกเซล
- **จำนวนคลาส:** **{len(class_names)}** คลาส
- **การกระจายข้อมูล:** แต่ละคลาสมีจำนวนภาพที่สมดุลกัน (Balanced Dataset) โดยมี **6,000 ภาพ** ในชุด Train และ **1,000 ภาพ** ในชุด Test
""")

st.subheader("10 คลาสในการจำแนก (Labels)")
class_list = ""
for i, name in enumerate(class_names):
    class_list += f"- **Label {i}:** {name}\n"
st.markdown(class_list)

# --- ส่วนโค้ดที่ถูกแก้ไข: แสดง 10 คลาสแบบไม่สุ่ม ---

# 1. ค้นหา Index ของตัวอย่างแรกสำหรับแต่ละคลาส (0 ถึง 9)
sample_indices = []
for class_index in range(len(class_names)):
    # np.where(y_train == class_index) จะคืนค่า Index ทั้งหมดที่ตรงกับ class_index
    # [0][0] คือการดึง Index ตัวแรกออกมา
    try:
        idx = np.where(y_train == class_index)[0][0]
        sample_indices.append(idx)
    except IndexError:
        # กรณีฉุกเฉิน: ถ้าหาคลาสไม่เจอ (ไม่น่าจะเกิดขึ้นกับ Fashion MNIST)
        st.warning(f"ไม่พบตัวอย่างสำหรับคลาส {class_index}")
        pass


st.subheader("ตัวอย่างข้อมูลจาก Dataset")
st.markdown("ตัวอย่าง 10 ภาพ (**1 ภาพต่อ 1 คลาส**):")
cols = st.columns(10)

# 2. วนลูปแสดงผลโดยใช้ Index ที่เตรียมไว้
for i, col in enumerate(cols):
    if i < len(sample_indices):
        idx_to_display = sample_indices[i]
        
        # แสดงภาพและชื่อคลาสที่ถูกต้อง
        col.image(x_train[idx_to_display], 
                  caption=class_names[y_train[idx_to_display]], 
                  width=70)
    else:
        break # ป้องกันการแสดงผลเกิน 10 คลาส


# Data preprocessing explanation
st.subheader("ขั้นตอนการเตรียมข้อมูลเบื้องต้น (Data Preprocessing)")
st.markdown("""
ก่อนนำข้อมูลเข้าสู่กระบวนการเทรนโมเดล (ทั้ง CNN และ Transfer Learning) ต้องมีการเตรียมการดังนี้:

1.  **Normalization (การปรับมาตรฐาน):** * ค่าพิกเซลเดิมอยู่ในช่วง **0 ถึง 255** จะถูกหารด้วย 255 เพื่อปรับให้อยู่ในช่วง **0.0 ถึง 1.0** * **วัตถุประสงค์:** ช่วยให้กระบวนการ Training มีความเสถียร, ลดความเหลื่อมล้ำของค่า Input, และช่วยให้โมเดลสามารถคอนเวอร์จ (Converge) ได้เร็วขึ้น

2.  **Categorical Encoding (การเข้ารหัสคลาส):** * Label (ตัวเลข 0-9) จะถูกแปลงเป็นรูปแบบ **One-Hot Vector** (เช่น คลาส 3 จะเป็น [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]) 
    * **วัตถุประสงค์:** เพื่อใช้กับ Loss Function ประเภท `categorical_crossentropy` ในการฝึกโมเดล

3.  **การปรับสำหรับ Transfer Learning (MobileNetV2):** * เนื่องจากโมเดล Pre-trained เช่น MobileNetV2 ถูกฝึกมากับภาพสีขนาดใหญ่ (เช่น ImageNet) ภาพ $28 \times 28$ ขาวดำจึงต้องถูก **Resize** ให้มีขนาดใหญ่ขึ้น (เช่น $96 \times 96$) และ **Replicate** เป็น 3 Channels (Stacking) เพื่อให้มีรูปร่างตรงตาม Input ที่โมเดลคาดหวัง
""")