# compare.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import CNNModel, HybridResNetEfficientNet, create_vit_model
from data_loader import create_dataloaders
from utils import get_device
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score  # Tính ROC-AUC

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Hàm thu thập dự đoán (nhãn thật và xác suất) từ mô hình
def get_model_predictions(model, dataloader, device):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            all_probs.append(probabilities.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    return all_labels, all_probs

# Hàm vẽ ROC curve trên một trục đã cho (ax)
def plot_multiclass_roc_on_ax(ax, all_labels, all_probs, num_classes, title="ROC Curve"):
    # Chuyển đổi nhãn thành dạng nhị phân (one-hot)
    all_labels_bin = label_binarize(all_labels, classes=range(num_classes))
    
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc_val:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2)  # Đường chéo (baseline)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

# Hàm đánh giá mô hình: tính Accuracy, Loss và tổng ROC-AUC (one-vs-rest)
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            probabilities = torch.softmax(outputs, dim=1)
            all_probs.append(probabilities.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except Exception as e:
        print(f"Lỗi khi tính ROC-AUC: {e}")
        roc_auc = None

    return accuracy, avg_loss, roc_auc

# Hàm so sánh các mô hình và vẽ 2 biểu đồ:
# - Biểu đồ bên trái: So sánh Accuracy và Loss của 3 mô hình
# - Biểu đồ bên phải: ROC curve của mô hình Hybrid
def compare_models(dataset_path):
    device = get_device()
    
    print("Loading dataset...")
    dataset = load_from_disk(dataset_path)
    _, _, test_loader, _, _, _ = create_dataloaders(dataset, batch_size=32)

    num_classes = len(dataset.features['label'].names)
    
    models = {
        "CNN": CNNModel(num_classes),
        "Hybrid": HybridResNetEfficientNet(num_classes),
        "ViT": create_vit_model(num_classes)
    }
    
    model_paths = {
        "CNN": "best_model_cnn.pth",
        "Hybrid": "best_model_hybrid.pth",
        "ViT": "best_model_vit.pth"
    }
    
    results = {}       # Lưu kết quả (accuracy, loss, roc_auc) của từng mô hình
    hybrid_predictions = None  # Lưu dự đoán của mô hình Hybrid cho ROC curve

    for model_name, model in models.items():
        model_path = model_paths.get(model_name)
        if not os.path.exists(model_path):
            print(f"Model file '{model_path}' not found! Skipping {model_name}.")
            continue
        
        print(f"Evaluating {model_name} model...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        
        acc, loss, roc_auc = evaluate_model(model, test_loader, device)
        results[model_name] = {"accuracy": acc, "loss": loss, "roc_auc": roc_auc}
        
        # Lưu kết quả dự đoán của mô hình Hybrid để vẽ ROC curve
        if model_name == "Hybrid":
            hybrid_predictions = get_model_predictions(model, test_loader, device)
    
    if not results:
        print("No models were evaluated. Please check your model files.")
        return

    # Tạo 1 figure với 2 biểu đồ: bên trái cho Accuracy & Loss, bên phải cho ROC của Hybrid
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Biểu đồ bên trái: So sánh Accuracy và Loss của 3 mô hình ---
    model_names_list = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in model_names_list]
    losses = [results[m]["loss"] for m in model_names_list]
    
    x = np.arange(len(model_names_list))
    width = 0.35

    # Vẽ grouped bar chart: 1 cặp bar (Accuracy, Loss) cho mỗi mô hình.
    # Lưu ý: Vì scale của Accuracy (0-1) và Loss (thường >1) khác nhau nên ta dùng hai trục y.
    bars_acc = ax_left.bar(x - width/2, accuracies, width, label="Accuracy", color='tab:blue')
    ax_left.set_ylabel("Accuracy", color='tab:blue')
    ax_left.tick_params(axis='y', labelcolor='tab:blue')
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(model_names_list)
    ax_left.set_ylim(0, 1.1)

    ax_left_twin = ax_left.twinx()
    bars_loss = ax_left_twin.bar(x + width/2, losses, width, label="Loss", color='tab:red')
    ax_left_twin.set_ylabel("Loss", color='tab:red')
    ax_left_twin.tick_params(axis='y', labelcolor='tab:red')
    ax_left_twin.set_ylim(0, max(losses)*1.2)
    
    ax_left.set_title("Model Comparison: Accuracy & Loss")
    # Kết hợp legend từ 2 trục
    bars = bars_acc + bars_loss
    labels = [bar.get_label() for bar in bars]
    ax_left.legend(bars, labels, loc="upper center")

    # --- Biểu đồ bên phải: ROC curve của mô hình Hybrid ---
    if hybrid_predictions is not None:
        hybrid_labels, hybrid_probs = hybrid_predictions
        plot_multiclass_roc_on_ax(ax_right, hybrid_labels, hybrid_probs, num_classes, title="ROC Curve for Hybrid Model")
    else:
        ax_right.text(0.5, 0.5, "Hybrid model predictions not available", 
                      horizontalalignment="center", verticalalignment="center")
        ax_right.set_title("ROC Curve for Hybrid Model")
    
    plt.tight_layout()
    plt.show()

