import os
from datasets import load_dataset, Dataset, concatenate_datasets, ClassLabel, Features, Image
from PIL import Image as PILImage

def load_and_prepare_dataset(extra_images_dir, high_heel_dir, processed_dataset_path='processed_dataset'):
    # Load dataset ban đầu từ Hugging Face
    dataset = load_dataset("Andyrasika/ShoeSandalBootimages")

    # Cập nhật nhãn mới
    new_labels = dataset['train'].features['label'].names + ["High Heel"]
    class_label = ClassLabel(names=new_labels)

    # Định nghĩa kiểu dữ liệu (Feature)
    features = Features({
        "image": Image(),
        "label": class_label
    })
    label_mapping = {label: i for i, label in enumerate(new_labels)}

    #Hàm load & resize ảnh 
    def load_image(img_path, size=(224, 224)):
        return PILImage.open(img_path).convert("RGB").resize(size)

    #Load ảnh từ thư mục bổ sung (Boot, Sandal, Shoe)
    extra_data = []
    for label in ["Boot", "Sandal", "Shoe"]:
        folder_path = os.path.join(extra_images_dir, label)
        if os.path.exists(folder_path):
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                    img_path = os.path.join(folder_path, img_name)
                    extra_data.append({
                        "image": load_image(img_path),
                        "label": label_mapping[label]
                    })

    extra_dataset = Dataset.from_list(extra_data).cast(features)

    # Load ảnh High Heel
    high_heel_data = [{
        "image": load_image(os.path.join(high_heel_dir, img)),
        "label": label_mapping["High Heel"]
    } for img in os.listdir(high_heel_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    high_heel_dataset = Dataset.from_list(high_heel_data).cast(features)


    # Merge toàn bộ dataset
    dataset["train"] = dataset["train"].cast(features)
    new_dataset = concatenate_datasets([dataset['train'], extra_dataset, high_heel_dataset])

    # Lưu dataset để sử dụng sau này 
    new_dataset.save_to_disk(processed_dataset_path)
    return new_dataset

def load_processed_dataset(processed_dataset_path='processed_dataset'):
    return Dataset.load_from_disk(processed_dataset_path)
