import os
os.environ['HF_HOME'] = 'D:/huggingface'
os.environ['TRANSFORMERS_CACHE'] = 'D:/huggingface_cache'

import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from text_generation import generate_caption  
from gradcam_visualize import apply_gradcam
from models import HybridResNetEfficientNet 

st.set_page_config(page_title="Chatbot Giày Dép 👟", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hàm dự đoán loại giày
def predict_shoe_type(image, model, device):
    transform = transforms.Compose([ 
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device, dtype=torch.float32)
    model.to(device) 
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)  # Dự đoán với mô hình
        predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# Load mô hình Hybrid từ checkpoint đã train
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridResNetEfficientNet(num_classes=5).to(device)
    checkpoint_path = "best_model_hybrid.pth"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"✅ Mô hình đã load thành công từ: {checkpoint_path}")
    except Exception as e:
        print(f"Lỗi khi load mô hình: {str(e)}")
    return model, device

hybrid_model, device = load_model()

st.title("👟 Chatbot Giày Dép - Nhận Diện & Mô Tả Sản Phẩm")

st.sidebar.header("🖼️ Tải lên ảnh giày của bạn")
uploaded_file = st.sidebar.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])

st.subheader("💬 Chatbot Phân Tích Hình Ảnh")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.sidebar.image(image, caption="📷 Ảnh đã tải lên", use_container_width=True)

    if st.sidebar.button("🔍 Phân tích ảnh"):
        with st.spinner("⏳ Đang xử lý..."):
            try:
                # 1️⃣ Dự đoán loại giày từ ảnh bằng mô hình Hybrid
                class_names = ['Boot', 'Sandal', 'Shoe', 'High Heel', 'Slipper']
                predicted_class = predict_shoe_type(image, hybrid_model, device)
                shoe_type = class_names[predicted_class]

                # 2️⃣ Sinh mô tả bằng BLIP-2
                image_path = "temp_uploaded_image.jpg"
                image.save(image_path)  # Lưu ảnh tạm
                caption = generate_caption(image_path)

                # 3️⃣ Áp dụng Grad-CAM++
                heatmap_path = apply_gradcam(image_path, hybrid_model, device)

                # Hiển thị kết quả
                st.write(f"**📌 Mô tả sản phẩm:** {caption}")
                st.write(f"**🔖 Loại giày nhận diện:** {shoe_type}")

                if heatmap_path:
                    st.image(heatmap_path, caption="🔍 Grad-CAM++: Model tập trung vào vùng nào", use_container_width=True)
                else:
                    st.error("Không thể tạo Grad-CAM++! Vui lòng thử lại.")
            except Exception as e:
                st.error(f"Lỗi khi phân tích ảnh: {str(e)}")
else:
    st.write("📥 Vui lòng tải lên ảnh giày để chatbot phân tích!")
