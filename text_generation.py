import os
os.environ['HF_HOME'] = 'D:/huggingface'
os.environ['TRANSFORMERS_CACHE'] = 'D:/huggingface_cache'

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Đang tải mô hình BLIP‑2...")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

if torch.cuda.is_available():
    # Sử dụng 8-bit cho GPU (không cần gọi model.to(device))
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", 
        load_in_8bit=True, 
        device_map="auto"
    )
else:
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    model.to(device)  # Chỉ chuyển model sang device khi không dùng 8-bit

model.eval()
print("Mô hình đã sẵn sàng.")

def generate_caption(image_path):
    """Generate a caption for an input image using BLIP‑2"""
    try:
        # Mở và chuyển đổi ảnh
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        inputs = inputs.to(device, dtype)
        
        with torch.no_grad():
            out = model.generate(**inputs)
        
        if out is None:
            raise ValueError("Output from model.generate is None")
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Lỗi trong generate_caption: {str(e)}")
        return None
if __name__ == '__main__':
    image_path = r"D:\code DAP\Ảnh test\Shoe\istockphoto-1350560575-612x612.jpg"
    
    caption = generate_caption(image_path)
    print(f"📝 Caption: {caption}")

