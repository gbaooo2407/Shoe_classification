import torch
from data_preparation import load_and_prepare_dataset, load_processed_dataset
from datasets import load_dataset
from data_loader import create_dataloaders
from models import CNNModel, HybridResNetEfficientNet, create_vit_model
from train import train_model
from utils import get_device, print_device_info
from exploratory import plot_random_images, print_dataset_statistics, print_average_image_size
import subprocess
import os
from compare import compare_models

def menu():
    menu = """
    ------------------------------------
        1. Util the device (Using GPU instead)
        2. Load the dataset (Combining new dataset)
        3. Exploraty about the data
        4. Creating dataloader
        5. Training with CNN Model
        6. Training with Hybrid Model (ResNet 18 + EfficientNet B0)
        7. Training with ViT Model
        8. Compare each models
        9. Deployment with app
        10. Exit
    ------------------------------------"""
    print(menu)
    selection = int(input("Please input the function you want: "))
    return selection

def func1():
    device = get_device()
    print_device_info()
    return device

def func2():
    processed_dataset_path = "processed_dataset"  # Đường dẫn file dataset đã xử lý
    high_heel_dir = r"D:\code DAP\High Heel"
    extra_images_dir = r"D:\code DAP"  # Chứa các folder Boot, Sandal, Shoe

    # Kiểm tra xem dataset đã được xử lý trước đó chưa
    if os.path.exists(processed_dataset_path):
        print("Loading processed dataset...")
        dataset = load_processed_dataset(processed_dataset_path)
    else:
        print("Processing and saving dataset for the first time...")
        dataset = load_and_prepare_dataset(extra_images_dir, high_heel_dir, processed_dataset_path)
    return dataset

def func3(dataset):
    while True:
        print("""Please choose the function:
                1. Show random image
                2. Print dataset statistic
                3. Print average size of images
                4. Exit
            """)
        try:
            choose = int(input("Enter the number: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        if choose == 1:
            plot_random_images(dataset=dataset)
        elif choose == 2:
            print_dataset_statistics(dataset=dataset)
        elif choose == 3:
            print_average_image_size(dataset=dataset)
        elif choose == 4:
            break
        else:
            print("Invalid. Please try again")

def func4(dataset=None):
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(dataset, batch_size=32)
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

def func5(num_classes, device, train_loader, val_loader):
    cnn_model = CNNModel(num_classes).to(device)
    train_model(cnn_model, train_loader, val_loader, device, num_epochs=30,
                 best_model_path="best_model_cnn.pth")

def func6(num_classes, device, train_loader, val_loader):
    hybrid_model = HybridResNetEfficientNet(num_classes).to(device)
    train_model(hybrid_model, train_loader, val_loader, device, num_epochs=30,
                 best_model_path="best_model_hybrid.pth")

def func7(num_classes, device, train_loader, val_loader):
    vit_model = create_vit_model(num_classes).to(device)
    train_model(vit_model, train_loader, val_loader, device, num_epochs=30,
                     best_model_path="best_model_vit.pth")



def func8():
    try:
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error launching Streamlit app:", e)

def main():
    model = None
    device = None
    dataset = None
    # Sửa dòng khởi tạo các biến dưới đây:
    train_loader = None
    val_loader = None
    test_loader = None
    train_dataset = None
    val_dataset = None
    test_dataset = None

    while True:
        try:
            selection = menu()
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        if selection == 1:
            device = func1()

        elif selection == 2:
            dataset = func2()
            print('\nLoad dataset successfully')

        elif selection == 3:
            if dataset is None:
                print("Please load dataset first!")
            else:
                func3(dataset)

        elif selection == 4:
            if dataset is None:
                print("Please load dataset first!")
            else:
                train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = func4(dataset)

        elif selection == 5:
            if dataset is None or device is None or train_loader is None or val_loader is None:
                print("Please load dataset and initialize device first!")
                continue
            num_classes = len(dataset.features['label'].names)
            func5(num_classes, device, train_loader, val_loader)

        elif selection == 6:
            if dataset is None or device is None or train_loader is None or val_loader is None:
                print("Please load dataset and initialize device first!")
                continue
            num_classes = len(dataset.features['label'].names)
            func6(num_classes, device, train_loader, val_loader)

        elif selection == 7:
            if dataset is None or device is None or train_loader is None or val_loader is None:
                print("Please load dataset and initialize device first!")
                continue
            num_classes = len(dataset.features['label'].names)
            func7(num_classes, device, train_loader, val_loader)

        elif selection == 8:
            compare_models('processed_dataset')

        elif selection ==9:
            func8()

        elif selection == 10:
            print("Exiting the application.")
            break

        else:
            print("Invalid option. Please choose a valid option.")

if __name__ == "__main__":
    main()
