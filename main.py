import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import PennFudanDataset
from model import SimpleObjectDetector
from train_utils import train_one_epoch, collate_fn
from visualize import visualize_predictions
import urllib.request
import zipfile
import os
import torch.optim as optim

def download_and_extract_pennfudan():
    url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
    zip_path = "PennFudanPed.zip"
    extract_path = "PennFudanPed"
    if not os.path.exists(extract_path):
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        os.remove(zip_path)
    return extract_path

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset_path = download_and_extract_pennfudan()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = PennFudanDataset(dataset_path, transforms=transform, target_size=(256, 256))
    indices = torch.randperm(len(dataset)).tolist()
    split = int(0.8 * len(dataset))
    dataset_train = torch.utils.data.Subset(dataset, indices[:split])
    dataset_val = torch.utils.data.Subset(dataset, indices[split:])
    data_loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True, collate_fn=collate_fn)
    data_loader_val = DataLoader(dataset_val, batch_size=4, shuffle=False, collate_fn=collate_fn)
    num_classes = 2
    model = SimpleObjectDetector(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=20)
        print(f"Epoch {epoch} completed.")
    visualize_predictions(model, data_loader_val, device)

if __name__ == "__main__":
    main()
