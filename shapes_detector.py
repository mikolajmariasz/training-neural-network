import os
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Configuration
IMAGE_SIZE = (256, 256)
GRID_SIZE = 16  # After downsampling by factor of 16
NUM_CLASSES = 3  # circle, square, triangle
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_IMAGES_DIR = 'dataset/train/images'
TRAIN_LABELS_DIR = 'dataset/train/labels'
VAL_IMAGES_DIR = 'dataset/val/images'
VAL_LABELS_DIR = 'dataset/val/labels'

CLASSES = ['circle', 'square', 'triangle']
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

def load_dataset(image_dir, label_dir):
    images = sorted(os.listdir(image_dir))
    labels = sorted(os.listdir(label_dir))
    data = list(zip(images, labels))
    return data

class ShapesDataset(Dataset):
    def __init__(self, data, image_dir, label_dir):
        self.data = data
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.cell_w = IMAGE_SIZE[0] / GRID_SIZE
        self.cell_h = IMAGE_SIZE[1] / GRID_SIZE

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_file, label_file = self.data[idx]
        img_path = os.path.join(self.image_dir, img_file)
        lbl_path = os.path.join(self.label_dir, label_file)
        
        img = Image.open(img_path).convert('RGB')
        img = img.resize(IMAGE_SIZE)
        img = np.array(img) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32)

        target = self.create_target(lbl_path)
        return img, target

    def create_target(self, label_path):
        target = np.zeros((GRID_SIZE, GRID_SIZE, 1+4+NUM_CLASSES), dtype=np.float32)
        with open(label_path, 'r') as f:
            data = json.load(f)
        objects = data['objects']
        
        for obj in objects:
            bbox = obj['bbox']
            x_min, y_min, x_max, y_max = bbox
            w = x_max - x_min
            h = y_max - y_min
            cx = x_min + w/2.0
            cy = y_min + h/2.0
            
            cell_x = int(cx // self.cell_w)
            cell_y = int(cy // self.cell_h)
            
            if cell_x >= GRID_SIZE:
                cell_x = GRID_SIZE - 1
            if cell_y >= GRID_SIZE:
                cell_y = GRID_SIZE - 1
            
            rel_cx = (cx % self.cell_w) / self.cell_w
            rel_cy = (cy % self.cell_h) / self.cell_h
            rel_w = w / IMAGE_SIZE[0]
            rel_h = h / IMAGE_SIZE[1]
            
            class_id = CLASS_TO_ID[obj['label']]
            target[cell_y, cell_x, 0] = 1.0
            target[cell_y, cell_x, 1:5] = [rel_cx, rel_cy, rel_w, rel_h]
            target[cell_y, cell_x, 5+class_id] = 1.0
        
        return torch.tensor(target, dtype=torch.float32)

class SimpleDetector(nn.Module):
    def __init__(self, in_channels=3, grid_size=GRID_SIZE, num_classes=NUM_CLASSES):
        super(SimpleDetector, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x128
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )
        self.pred = nn.Conv2d(256, (1+4+num_classes), kernel_size=1, stride=1)
        self.grid_size = grid_size
        self.num_classes = num_classes
    
    def forward(self, x):
        x = self.features(x)
        x = self.pred(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

def detection_loss(y_pred, y_true):
    obj_true = y_true[..., 0]
    box_true = y_true[..., 1:5]
    class_true = y_true[..., 5:]
    
    obj_pred = y_pred[..., 0]
    box_pred = y_pred[..., 1:5]
    class_pred = y_pred[..., 5:]
    
    obj_loss = nn.BCEWithLogitsLoss()(obj_pred, obj_true)

    obj_mask = obj_true.unsqueeze(-1)
    box_loss = (obj_mask * (box_true - box_pred)**2).sum() / (obj_mask.sum() + 1e-6)

    class_indices = class_true.argmax(dim=-1)
    class_mask = (obj_true == 1)
    if class_mask.sum() > 0:
        class_pred_filtered = class_pred[class_mask]
        class_indices_filtered = class_indices[class_mask]
        class_loss = nn.CrossEntropyLoss()(class_pred_filtered, class_indices_filtered)
    else:
        class_loss = torch.tensor(0.0, device=DEVICE)
    
    total_loss = obj_loss + box_loss + class_loss
    return total_loss, obj_loss, box_loss, class_loss

def train_model(epochs=20):
    train_data = load_dataset(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
    val_data = load_dataset(VAL_IMAGES_DIR, VAL_LABELS_DIR)
    train_dataset = ShapesDataset(train_data, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
    val_dataset = ShapesDataset(val_data, VAL_IMAGES_DIR, VAL_LABELS_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    model = SimpleDetector().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        model.train()
        total_loss_val = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            preds = model(imgs)
            loss, _, _, _ = detection_loss(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss_val += loss.item()
        
        avg_train_loss = total_loss_val / len(train_loader)
        
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(DEVICE)
                targets = targets.to(DEVICE)
                preds = model(imgs)
                loss, _, _, _ = detection_loss(preds, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), "model_256x256.pth")
    print("Model saved to model_256x256.pth")

def decode_predictions(preds, conf_threshold=0.5):
    obj_scores = torch.sigmoid(preds[..., 0])
    box_coords = preds[..., 1:5]
    class_logits = preds[..., 5:]
    class_probs = F.softmax(class_logits, dim=-1)
    
    detections = []
    cell_w = IMAGE_SIZE[0] / GRID_SIZE
    cell_h = IMAGE_SIZE[1] / GRID_SIZE
    
    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            score = obj_scores[gy, gx].item()
            if score > conf_threshold:
                cx, cy, w, h = box_coords[gy, gx].tolist()
                box_cx = gx * cell_w + cx * cell_w
                box_cy = gy * cell_h + cy * cell_h
                box_w = w * IMAGE_SIZE[0]
                box_h = h * IMAGE_SIZE[1]
                x_min = box_cx - box_w/2
                y_min = box_cy - box_h/2
                x_max = box_cx + box_w/2
                y_max = box_cy + box_h/2
                
                class_conf, class_id = torch.max(class_probs[gy, gx], dim=-1)
                class_conf = class_conf.item()
                class_id = class_id.item()
                final_conf = score * class_conf
                detections.append({
                    'bbox': [x_min, y_min, x_max, y_max],
                    'class_id': class_id,
                    'score': final_conf
                })
    return detections

def visualize_predictions(num_images=5, conf_threshold=0.5):
    val_data = load_dataset(VAL_IMAGES_DIR, VAL_LABELS_DIR)
    model = SimpleDetector().to(DEVICE)
    model.load_state_dict(torch.load("model_256x256.pth", map_location=DEVICE))
    model.eval()

    for i in range(num_images):
        img_file, _ = val_data[i]
        img_path = os.path.join(VAL_IMAGES_DIR, img_file)
        img_pil = Image.open(img_path).convert('RGB')
        img_pil_resized = img_pil.resize(IMAGE_SIZE)
        img_np = np.array(img_pil_resized) / 255.0
        img_tensor = torch.from_numpy(np.transpose(img_np, (2,0,1))).float().unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            preds = model(img_tensor)
        preds = preds[0].cpu()
        
        detections = decode_predictions(preds, conf_threshold=conf_threshold)
        
        fig, ax = plt.subplots(1)
        ax.imshow(img_pil_resized)
        for det in detections:
            x_min, y_min, x_max, y_max = det['bbox']
            class_id = det['class_id']
            score = det['score']
            label = f"{CLASSES[class_id]}: {score:.2f}"
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x_min, y_min - 5, label, color='red', fontsize=12, backgroundcolor='white')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Shape Detector (256x256)')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'visualize'], help='Mode: train or visualize')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--num_images', type=int, default=5, help='Number of images to visualize')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='Confidence threshold for visualization')
    args = parser.parse_args()

    if args.mode == 'train':
        train_model(epochs=args.epochs)
    elif args.mode == 'visualize':
        visualize_predictions(num_images=args.num_images, conf_threshold=args.conf_threshold)

if __name__ == "__main__":
    main()
