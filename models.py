# models.py
import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet
from timm import create_model

# Mô hình CNN đơn giản
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.flatten = nn.Flatten()
        # Lưu ý: Kích thước đầu vào của fc1 cần điều chỉnh theo kích thước ảnh sau khi qua các layer convolution
        self.fc1 = nn.Linear(64 * 54 * 54, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Mô hình Hybrid kết hợp ResNet-18 và EfficientNet-B0 và LSTM
class HybridResNetEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(HybridResNetEfficientNet, self).__init__()
        # ResNet-18 branch
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Lấy feature vector có kích thước 512
        # EfficientNet-B0 branch
        self.effnet = EfficientNet.from_pretrained('efficientnet-b0')
        # Classifier kết hợp: 512 + 1280 = 1792
        self.classifier = nn.Sequential(
            nn.Linear(512 + 1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feat_resnet = self.resnet(x)  # [batch, 512]
        feat_effnet = self.effnet.extract_features(x)  # [batch, C, H, W]
        feat_effnet = self.effnet._avg_pooling(feat_effnet)  # [batch, 1280, 1, 1]
        feat_effnet = feat_effnet.flatten(1)  # [batch, 1280]
        features = torch.cat((feat_resnet, feat_effnet), dim=1)  # [batch, 1792]
        out = self.classifier(features)
        return out



# Hàm tạo mô hình ViT sử dụng timm
def create_vit_model(num_classes):
    model_vit = create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)
    return model_vit
