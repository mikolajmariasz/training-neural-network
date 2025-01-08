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
from collections import defaultdict
import time

# Configuration
IMAGE_SIZE = (256, 256)
GRID_SIZE = 16
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_IMAGES_DIR = 'dataset/train/images'
TRAIN_LABELS_DIR = 'dataset/train/labels'
VAL_IMAGES_DIR = 'dataset/val/images'
VAL_LABELS_DIR = 'dataset/val/labels'

CLASSES = ['star', 'hexagon', 'arrow', 'circle', 'square', 'triangle']
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES) 

IOU_THRESHOLD = 0.5

os.makedirs("logs", exist_ok=True)

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
        img_tensor = torch.tensor(img, dtype=torch.float32)

        target = self.create_target(lbl_path)
        return img_tensor, target

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

############################
# Model Architectures
############################

class ModelSmall(nn.Module):
    def __init__(self, in_channels=3):
        super(ModelSmall, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pred = nn.Conv2d(128, (1+4+NUM_CLASSES), kernel_size=1, stride=1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.pred(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

class ModelMedium(nn.Module):
    def __init__(self, in_channels=3):
        super(ModelMedium, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128,256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pred = nn.Conv2d(256, (1+4+NUM_CLASSES), kernel_size=1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.pred(x)
        x = x.permute(0,2,3,1)
        return x

class ModelLarge(nn.Module):
    def __init__(self, in_channels=3):
        super(ModelLarge, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128,128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128,256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256,256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pred = nn.Conv2d(256, (1+4+NUM_CLASSES), kernel_size=1)
    def forward(self, x):
        x = self.features(x)
        x = self.pred(x)
        x = x.permute(0,2,3,1)
        return x

def get_model(architecture):
    if architecture == 'Model 1':
        return ModelSmall().to(DEVICE)
    elif architecture == 'Model 2':
        return ModelMedium().to(DEVICE)
    elif architecture == 'Model 3':
        return ModelLarge().to(DEVICE)
    else:
        raise ValueError("Unknown architecture")

############################
# Loss and Training
############################
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

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea / float(boxAArea+boxBArea-interArea)

def nms(detections, iou_threshold=0.5):
    result = []
    for cls_id in range(NUM_CLASSES):
        cls_dets = [d for d in detections if d['class_id'] == cls_id]
        cls_dets.sort(key=lambda x: x['score'], reverse=True)
        
        keep = []
        while cls_dets:
            best = cls_dets.pop(0)
            keep.append(best)
            cls_dets = [d for d in cls_dets if compute_iou(best['bbox'], d['bbox']) < iou_threshold]
        result.extend(keep)
    return result

def evaluate_map(model_path):
    val_data = load_dataset(VAL_IMAGES_DIR, VAL_LABELS_DIR)
    val_dataset = ShapesDataset(val_data, VAL_IMAGES_DIR, VAL_LABELS_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)
    
    parts = model_path.split('_')
    # expecting pattern: model_256x256_{arch}_best.pth
    # arch should be parts[-2]
    architecture = parts[-2]
    model = get_model(architecture)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    gt_boxes_per_class = defaultdict(list)
    pred_boxes = defaultdict(list)

    with torch.no_grad():
        for i, (imgs, targets) in enumerate(val_loader):
            imgs = imgs.to(DEVICE)
            preds = model(imgs) # (1,G,G,1+4+classes)
            preds = preds[0].cpu()
            detections = decode_predictions(preds, conf_threshold=0.05)  
            detections = nms(detections, iou_threshold=0.5)

            t = targets[0].numpy()
            for gy in range(GRID_SIZE):
                for gx in range(GRID_SIZE):
                    if t[gy,gx,0] == 1:
                        cx, cy, w, h = t[gy,gx,1:5]
                        cell_w = IMAGE_SIZE[0]/GRID_SIZE
                        cell_h = IMAGE_SIZE[1]/GRID_SIZE
                        box_cx = gx*cell_w + cx*cell_w
                        box_cy = gy*cell_h + cy*cell_h
                        box_w = w*IMAGE_SIZE[0]
                        box_h = h*IMAGE_SIZE[1]
                        x_min = box_cx - box_w/2
                        y_min = box_cy - box_h/2
                        x_max = box_cx + box_w/2
                        y_max = box_cy + box_h/2
                        
                        class_id = np.argmax(t[gy,gx,5:])
                        gt_boxes_per_class[class_id].append((i, [x_min,y_min,x_max,y_max]))
            
            for det in detections:
                c = det['class_id']
                pred_boxes[c].append((i, det['score'], det['bbox']))

    APs = []
    for c in range(NUM_CLASSES):
        pred_boxes[c].sort(key=lambda x:x[1], reverse=True)
        
        gt_boxes = gt_boxes_per_class[c]
        image_gt_map = defaultdict(list)
        for (img_id, box) in gt_boxes:
            image_gt_map[img_id].append(box)
        
        gt_matched = {img_id: [False]*len(boxes) for img_id, boxes in image_gt_map.items()}
        
        TP = []
        FP = []
        for (img_id, score, pbox) in pred_boxes[c]:
            best_iou = 0
            best_idx = -1
            if img_id in image_gt_map:
                for gt_i, gt_box in enumerate(image_gt_map[img_id]):
                    iou = compute_iou(pbox, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = gt_i

            if best_iou >= IOU_THRESHOLD and not gt_matched[img_id][best_idx]:
                TP.append(1)
                FP.append(0)
                gt_matched[img_id][best_idx] = True
            else:
                TP.append(0)
                FP.append(1)

        if len(TP) == 0:
            APs.append(0.0)
            continue
        
        TP = np.cumsum(TP)
        FP = np.cumsum(FP)
        total_gt = len(gt_boxes)
        recall = TP / (total_gt+1e-6)
        precision = TP / (TP+FP+1e-6)

        ap = 0.0
        for r_threshold in np.linspace(0,1,11):
            p = precision[recall >= r_threshold]
            if len(p) > 0:
                ap += np.max(p)
            else:
                ap += 0.0
        ap = ap / 11.0
        APs.append(ap)
    
    mAP = np.mean(APs) if len(APs)>0 else 0.0
    return mAP

def log_training(arch, train_losses, val_losses, best_val_loss, total_time, mAP_score):
    log_path = f"logs/log_{arch}.txt"
    with open(log_path, 'w') as f:
        f.write(f"Model Architecture: {arch}\n")
        f.write(f"Training time (seconds): {total_time:.2f}\n")
        f.write("Epoch,TrainLoss,ValLoss\n")
        for i, (tl, vl) in enumerate(zip(train_losses, val_losses)):
            f.write(f"{i+1},{tl:.4f},{vl:.4f}\n")
        f.write(f"Best Val Loss: {best_val_loss:.4f}\n")
        f.write(f"mAP: {mAP_score:.4f}\n")

def run_experiment(epochs=20, early_stop_patience=3):
    architectures = ['Model 1', 'Model 2', 'Model 3']
    mAP_results = {}
    for arch in architectures:
        train_data = load_dataset(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
        val_data = load_dataset(VAL_IMAGES_DIR, VAL_LABELS_DIR)
        train_dataset = ShapesDataset(train_data, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
        val_dataset = ShapesDataset(val_data, VAL_IMAGES_DIR, VAL_LABELS_DIR)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

        model = get_model(arch)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

        print(f"Training {arch} model...")
        start_time = time.time()  # Start timer
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
            train_losses.append(avg_train_loss)

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
            val_losses.append(avg_val_loss)

            print(f"Architecture: {arch}, Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"model_256x256_{arch}_best.pth")
            else:
                patience_counter += 1
            
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Training time for {arch} model: {total_time:.2f} seconds")

        best_epoch = np.argmin(val_losses)
        plt.figure()
        plt.plot(train_losses, label='Strata treningowa')
        plt.plot(val_losses, label='Strata walidacyjna')
        plt.scatter(best_epoch, val_losses[best_epoch], color='red', label='Najlepsza strata walidacyjna')
        plt.title(f"Historia strat - {arch}")
        plt.xlabel('Epoka')
        plt.ylabel('Strata')
        plt.legend()
        plt.savefig(f'loss_history_{arch}.png')
        plt.close()

        best_model_path = f"model_256x256_{arch}_best.pth"
        mAP_score = evaluate_map(best_model_path)
        mAP_results[arch] = mAP_score
        print(f"Architecture: {arch}, mAP: {mAP_score:.4f}")
        
        log_training(arch, train_losses, val_losses, best_val_loss, total_time, mAP_score)

    # Plot mAP comparison
    architectures = list(mAP_results.keys())
    mAP_values = [mAP_results[a] for a in architectures]
    plt.figure()
    plt.bar(architectures, mAP_values, color=['blue','green','orange'])
    plt.title("Porównanie mAP")
    plt.ylabel("mAP")
    for i, v in enumerate(mAP_values):
        plt.text(i, v+0.01, f"{v:.3f}", ha='center')
    plt.savefig("mAP_comparison.png")
    plt.close()


def test_predictions():
    val_data = load_dataset(VAL_IMAGES_DIR, VAL_LABELS_DIR)
    sample_indices = [8,9,10]
    img_paths = [os.path.join(VAL_IMAGES_DIR, val_data[i][0]) for i in sample_indices]

    architectures = ['Model 1', 'Model 2', 'Model 3']
    models = {}
    for arch in architectures:
        model_path = f"model_256x256_{arch}_best.pth"
        model = get_model(arch)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        models[arch] = model

    fig, axes = plt.subplots(len(sample_indices), len(architectures)+1, figsize=(15, 10))
    if len(sample_indices) == 1:
        axes = [axes]  

    for row_idx, idx in enumerate(sample_indices):
        img_pil = Image.open(img_paths[row_idx]).convert('RGB')
        img_pil_resized = img_pil.resize(IMAGE_SIZE)
        img_np = np.array(img_pil_resized) / 255.0
        img_tensor = torch.from_numpy(np.transpose(img_np, (2,0,1))).float().unsqueeze(0).to(DEVICE)

        # Show original image
        axes[row_idx][0].imshow(img_pil_resized)
        axes[row_idx][0].set_title("Oryginał")
        axes[row_idx][0].axis('off')

        col_idx = 1
        for arch in architectures:
            with torch.no_grad():
                preds = models[arch](img_tensor)
            preds = preds[0].cpu()
            detections = decode_predictions(preds, conf_threshold=0.5)
            detections = nms(detections, 0.5)

            axes[row_idx][col_idx].imshow(img_pil_resized)
            for det in detections:
                x_min, y_min, x_max, y_max = det['bbox']
                class_id = det['class_id']
                score = det['score']
                label = f"{CLASSES[class_id]}: {score:.2f}"
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                    fill=False, color='red', linewidth=2)
                axes[row_idx][col_idx].add_patch(rect)
                axes[row_idx][col_idx].text(x_min, y_min - 5, label, color='red', fontsize=12, backgroundcolor='white')
            axes[row_idx][col_idx].set_title(f"Predykcje - {arch}")
            axes[row_idx][col_idx].axis('off')
            col_idx += 1

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Shape Detector with multiple models, mAP, timing, and test predictions')
    parser.add_argument('--mode', type=str, required=True, choices=['train','experiment','test'],
                        help='Mode: train single architecture, run experiment with all three, or test predictions')
    parser.add_argument('--architecture', type=str, choices=['small','medium','large'], default='small',
                        help='Model architecture to use when mode=train')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--early_stop_patience', type=int, default=5, 
                    help='Number of epochs to wait without improvement before stopping training')
    args = parser.parse_args()

    if args.mode == 'train':
        # Train single architecture
        train_data = load_dataset(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
        val_data = load_dataset(VAL_IMAGES_DIR, VAL_LABELS_DIR)
        train_dataset = ShapesDataset(train_data, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
        val_dataset = ShapesDataset(val_data, VAL_IMAGES_DIR, VAL_LABELS_DIR)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

        model = get_model(args.architecture)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

        patience_counter = 0


        print(f"Training {args.architecture} model...")
        start_time = time.time()
        for epoch in range(args.epochs):
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
            train_losses.append(avg_train_loss)

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
            val_losses.append(avg_val_loss)

            print(f"Architecture: {args.architecture}, Epoch [{epoch+1}/{args.epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"model_{args.architecture}_best.pth")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Training time for {args.architecture} model: {total_time:.2f} seconds")

        best_epoch = np.argmin(val_losses)
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.scatter(best_epoch, val_losses[best_epoch], color='red', label='Best Val Loss')
        plt.title(f"Loss history - {args.architecture} model")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'loss_history_{args.architecture}.png')
        plt.close()

        best_model_path = f"model_256x256_{args.architecture}_best.pth"
        mAP_score = evaluate_map(best_model_path)
        print(f"Architecture: {args.architecture}, mAP: {mAP_score:.4f}")

    elif args.mode == 'experiment':
        # Train all three models, plot results and mAP comparison
        run_experiment(epochs=args.epochs, early_stop_patience=args.early_stop_patience)
    elif args.mode == 'test':
        # Test predictions on a fixed set of images for all three models
        test_predictions()
        

if __name__ == "__main__":
    main()
