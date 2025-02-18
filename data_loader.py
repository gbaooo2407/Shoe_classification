# data_loader.py
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image as PILImage

# Định nghĩa transform (điều chỉnh kích thước, augmentation, normalize,...)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize ảnh về 224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        """
        dataset: một danh sách hoặc một Subset có chứa các phần tử kiểu dict với key "image" và "label"
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]  # dict: {'image': image, 'label': label}
        image, label = sample["image"], sample["label"]
        if self.transform:
            image = self.transform(image)
        return image, label

def create_dataloaders(dataset, batch_size=32, test_size=0.1, val_size_ratio=20/90):
    """
    Chia dữ liệu train thành train, validation và test
    """
    indices = np.arange(len(dataset))
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=42
    )
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size_ratio,
        random_state=42
    )
    # Ép kiểu chỉ số thành int
    train_indices = [int(idx) for idx in train_indices]
    val_indices = [int(idx) for idx in val_indices]
    test_indices = [int(idx) for idx in test_indices]

    # Tạo Subset cho từng phần
    train_data = Subset(dataset, train_indices)
    val_data = Subset(dataset, val_indices)
    test_data = Subset(dataset, test_indices)

    # Tạo các dataset với custom transform
    train_dataset = CustomDataset(dataset=train_data, transform=transform)
    val_dataset = CustomDataset(dataset=val_data, transform=transform)
    test_dataset = CustomDataset(dataset=test_data, transform=transform)

    # Tạo DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
