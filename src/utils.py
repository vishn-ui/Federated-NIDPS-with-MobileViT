import torch
from torchvision import datasets, transforms
import os

def get_dataloader(partition="train"):
    data_path = f"/mnt/c/Users/vishn/Desktop/Malware_Project/DetectionDataset/DetectionDataset/splittedDataset/{partition}"
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    dataset = datasets.ImageFolder(data_path, transform=transform)
    # Note: Images are converted to Float64 inside the training loop
    return torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)