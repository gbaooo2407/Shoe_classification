# utils.py
import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_device_info():
    device = get_device()
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected")
