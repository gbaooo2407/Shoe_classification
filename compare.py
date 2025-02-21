import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import CNNModel, HybridResNetEfficientNet, create_vit_model
from data_loader import create_dataloaders
from utils import get_device
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score, f1_score

def evaluate_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
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
    
    pred_labels = np.argmax(all_probs, axis=1)
    f1 = f1_score(all_labels, pred_labels, average="weighted")
    
    return accuracy, avg_loss, f1, roc_auc

def compare_models(dataset_path):
    device = get_device()
    dataset = load_from_disk(dataset_path)
    _, _, test_loader, _, _, _ = create_dataloaders(dataset, batch_size=32)
    num_classes = len(dataset.features['label'].names)
    
    models_dict = {
        "CNN": CNNModel(num_classes),
        "Hybrid": HybridResNetEfficientNet(num_classes),
        "ViT": create_vit_model(num_classes)
    }
    
    model_paths = {
        "CNN": "best_model_cnn.pth",
        "Hybrid": "best_model_hybrid.pth",
        "ViT": "best_model_vit.pth"
    }
    
    results = {}
    
    for model_name, model in models_dict.items():
        model_path = model_paths.get(model_name)
        if not os.path.exists(model_path):
            print(f"Model file '{model_path}' not found! Skipping {model_name}.")
            continue
        
        print(f"Evaluating {model_name} model...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        
        acc, loss, f1, roc_auc = evaluate_model(model, test_loader, device)
        results[model_name] = {"accuracy": acc, "loss": loss, "f1": f1, "roc_auc": roc_auc}
    
    if not results:
        print("No models were evaluated. Please check your model files.")
        return
    
    # In kết quả của từng mô hình
    for model, metrics in results.items():
        print(f"{model} - Accuracy: {metrics['accuracy']:.4f}, Loss: {metrics['loss']:.4f}, "
              f"F1-Score: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Vẽ biểu đồ bar gộp cả 4 chỉ số
    model_names_list = list(results.keys())
    metrics_labels = ["Accuracy", "Loss", "F1-Score", "ROC-AUC"]
    data = np.array([[results[m]["accuracy"], results[m]["loss"], results[m]["f1"], results[m]["roc_auc"]] 
                      for m in model_names_list])
    
    x = np.arange(len(model_names_list))
    width = 0.2  # Độ rộng của mỗi thanh
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['skyblue', 'salmon', 'lightgreen', 'plum']
    
    for i in range(4):  # 4 chỉ số
        ax.bar(x + (i - 1.5) * width, data[:, i], width, label=metrics_labels[i], color=colors[i])
    
    ax.set_xticks(x)
    ax.set_xticklabels(model_names_list)
    ax.set_ylabel("Metric Values")
    ax.set_title("Model Comparison: Accuracy, Loss, F1-Score, and ROC-AUC")
    ax.legend()
    plt.show()
