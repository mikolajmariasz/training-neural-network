# main.py

# -----
# TODO: Implement transforms for dataset augumentation
# -----

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import PennFudanDataset
from model import SimpleObjectDetector
from train_utils import train_one_epoch, collate_fn
from visualize import visualize_predictions, visualize_initial_predictions
import urllib.request
import zipfile
import os
import torch.optim as optim

def download_and_extract_pennfudan(): 
    """
    Function to download and extract dataset

    Returns:
        extract_path (string): Path to loc where dataset was extracted
    """
    url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
    zip_path = "PennFudanPed.zip"
    extract_path = "PennFudanPed"
    # Check if dataset was already extracted
    if not os.path.exists(extract_path):
        print("Downloading Penn-Fudan dataset...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        os.remove(zip_path) # Remove zipped dataset
    return extract_path

BATCH_SIZE = 16  # Number of images in one batch
NUM_PREDICTIONS = 5  # Number of potential bounding boxes on image, TODO: Refactor code to be able to work with non-fixed amount of objects 
NUM_CLASSES = 2  # Background and pedestrian
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3

def main():
    """
    Main func operating training and evaluation process
    - Downloads and prepares the dataset
    - Initializes the model and optimizer
    - Visualizes evaluation images with initial predictions
    - Trains the model
    - Visualizes final results after training
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # If GPU is avaible switch from CPU
    print(f"Using device: {device}")
    dataset_path = download_and_extract_pennfudan()
    transform = transforms.Compose([ #  
        transforms.ToTensor(), # Converts images to pytorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalizes tensor with mean and standard deviation
    ])
    dataset = PennFudanDataset(dataset_path, transforms=transform, target_size=(256, 256)) # Dataset initialization
    indices = torch.randperm(len(dataset)).tolist() # Generates random permutation of images for psoitive training
    split = int(0.8 * len(dataset)) # Calculate split index
    dataset_train = torch.utils.data.Subset(dataset, indices[:split]) # Training subset (80%)
    dataset_val = torch.utils.data.Subset(dataset, indices[split:]) # Evaluation subset (20%)
    # Dataloaders for training and ev subsets
    data_loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    data_loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    model = SimpleObjectDetector(num_classes=NUM_CLASSES, num_predictions=NUM_PREDICTIONS).to(device) # Init model and move it to device
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Define optimizer (Adam) with model params and learning rate

    # Visualize initial preds before training
    print("Visualizing initial predictions before training...")
    visualize_initial_predictions(model, data_loader_val, device)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"Starting Epoch {epoch+1}/{NUM_EPOCHS}")
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=20)
        print(f"Epoch {epoch+1} completed.")
    
    print("Training completed. Visualizing predictions on validation set.")
    visualize_predictions(model, data_loader_val, device)

if __name__ == "__main__":
    main() # Exec main func
