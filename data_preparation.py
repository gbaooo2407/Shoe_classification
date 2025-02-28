import os
import pandas as pd
import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ZapposDataset(Dataset):
    def __init__(self, image_dir, meta_data_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Load metadata
        self.meta_data = pd.read_csv(meta_data_path)
        self.image_paths = self.meta_data['image_path'].tolist()
        self.descriptions = self.meta_data['description'].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        description = self.descriptions[idx]
        
        return image, description

