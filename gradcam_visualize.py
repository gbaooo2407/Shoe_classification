import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from PIL import Image

def apply_gradcam(image_path: str, model, device, class_names=None, target_layer=None):
    """
    Áp dụng Grad-CAM++ lên một ảnh duy nhất.
    
    Args:
        image_path (str): Đường dẫn tới ảnh đầu vào.
        model (torch.nn.Module): Mô hình đã được load.
        device (str): Thiết bị ('cuda' hoặc 'cpu').
        class_names (list, optional): Danh sách tên các lớp.
        target_layer (torch.nn.Module, optional): Lớp mục tiêu để tính Grad-CAM.
    
    Returns:
        str: Đường dẫn của ảnh heatmap đã được lưu.
    """
    model.eval()

    # Xác định target_layer nếu chưa có
    # model_name = type(model).__name__
    # if target_layer is None:
    #     if "HybridResNetEfficientNet" in model_name:
    #     elif "VisionTransformer" in model_name or "ViT" in model_name:
    #         target_layer = model.blocks[0].attn  # Sử dụng lớp attention đầu tiên của ViT
    #     elif "ResNet" in model_name:
    #         target_layer = list(model.children())[-2]  
    #     elif "CNNModel" in model_name:
    #         target_layer = model.conv2  
    #     else:
    #         raise ValueError(f"Unsupported model type: {model_name}")
    target_layer = model.effnet._conv_head  

    # Khởi tạo Grad-CAM++
    cam_extractor = GradCAMPlusPlus(model, [target_layer])
    
    # Tiền xử lý ảnh
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize ảnh về 224x224 cho ViT
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Mean, Std của ViT
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Dự đoán lớp
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax().item()

    targets = [ClassifierOutputTarget(pred_class)]
    
    # Tính Grad-CAM++
    cam_output = cam_extractor(input_tensor, targets=targets)
    heatmap = cam_output[0]

    # Chuẩn hóa heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # Chuyển ảnh từ PIL -> NumPy
    original_np = np.array(image)

    # Resize heatmap về kích thước ảnh gốc
    heatmap_resized = cv2.resize(heatmap, (original_np.shape[1], original_np.shape[0]))
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Chồng heatmap lên ảnh gốc
    original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
    overlayed_image = cv2.addWeighted(original_bgr, 0.7, heatmap_colored, 0.3, 0)
    overlayed_image = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)

    # Lưu ảnh
    heatmap_path = "gradcam_result.jpg"
    cv2.imwrite(heatmap_path, cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))
    
    return heatmap_path
