import streamlit as st

st.set_page_config(
    page_title="Fashion MNIST Deep Learning App",
    page_icon="🧥",
    layout="wide"
)

st.title("👗 Fashion MNIST Deep Learning Models Showcase")
st.markdown("""
ยินดีต้อนรับสู่เว็บแอปที่รวบรวมโมเดล Deep Learning 
ในการจำแนกรูปภาพจาก **Fashion-MNIST Dataset**  
เลือกหน้าทางด้านซ้ายเพื่อดูรายละเอียดแต่ละโมเดลได้เลย 👇
""")

st.image("https://github.com/zalandoresearch/fashion-mnist/raw/master/doc/img/fashion-mnist-sprite.png")
