import os
import random
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.ops import nms  
from tqdm import tqdm  
import matplotlib.pyplot as plt  


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


IMAGE_SIZE = 128
NUM_CLASSES = 3  
CLASS_NAMES = ['circle', 'square', 'triangle']
NUM_TRAIN_IMAGES = 5000  
NUM_VAL_IMAGES = 1000    
MAX_SHAPES_PER_IMAGE = 5
GRID_SIZE = 7  
ANCHORS_PER_CELL = 2  
BATCH_SIZE = 32
EPOCHS = 30  
LEARNING_RATE = 0.001
DATASET_DIR = 'dataset'
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'val')
TRAIN_IMAGES_DIR = os.path.join(TRAIN_DIR, 'images')
TRAIN_ANNOTATIONS_DIR = os.path.join(TRAIN_DIR, 'annotations')
VAL_IMAGES_DIR = os.path.join(VAL_DIR, 'images')
VAL_ANNOTATIONS_DIR = os.path.join(VAL_DIR, 'annotations')
IOU_THRESHOLD = 0.5  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_directories():
    """Create necessary directories for the dataset."""
    dirs = [
        TRAIN_IMAGES_DIR,
        TRAIN_ANNOTATIONS_DIR,
        VAL_IMAGES_DIR,
        VAL_ANNOTATIONS_DIR
    ]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    print(f"Dataset directories created at '{DATASET_DIR}'.")

def generate_and_save_synthetic_data(num_images, images_dir, annotations_dir):
    """Generate synthetic images with shapes and save them along with annotations."""
    for img_idx in tqdm(range(num_images), desc=f"Generating {os.path.basename(images_dir)}"):
        
        img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        num_shapes = random.randint(1, MAX_SHAPES_PER_IMAGE)
        boxes = []
        labels = []

        for _ in range(num_shapes):
            shape_class = random.randint(0, NUM_CLASSES - 1)
            color = tuple(np.random.randint(0, 256, size=3))
            
            for attempt in range(100):
                size = random.randint(10, 30)
                x_min = random.randint(0, IMAGE_SIZE - size - 1)
                y_min = random.randint(0, IMAGE_SIZE - size - 1)
                x_max = x_min + size
                y_max = y_min + size
                proposed_box = [x_min, y_min, x_max, y_max]

                
                overlap = False
                for box in boxes:
                    
                    inter_x_min = max(box[0], proposed_box[0])
                    inter_y_min = max(box[1], proposed_box[1])
                    inter_x_max = min(box[2], proposed_box[2])
                    inter_y_max = min(box[3], proposed_box[3])
                    inter_area = max(inter_x_max - inter_x_min, 0) * max(inter_y_max - inter_y_min, 0)
                    box_area = (box[2] - box[0]) * (box[3] - box[1])
                    proposed_area = (proposed_box[2] - proposed_box[0]) * (proposed_box[3] - proposed_box[1])
                    iou = inter_area / (box_area + proposed_area - inter_area) if (box_area + proposed_area - inter_area) > 0 else 0
                    if iou > 0.3:  
                        overlap = True
                        break
                if not overlap:
                    
                    if shape_class == 0:
                        
                        draw.ellipse(proposed_box, fill=color, outline=None)
                    elif shape_class == 1:
                        
                        draw.rectangle(proposed_box, fill=color, outline=None)
                    elif shape_class == 2:
                        
                        point1 = (x_min + size // 2, y_min)
                        point2 = (x_min, y_max)
                        point3 = (x_max, y_max)
                        draw.polygon([point1, point2, point3], fill=color, outline=None)
                    boxes.append(proposed_box)
                    labels.append(shape_class)
                    break  

        
        image_filename = f"img_{img_idx:05d}.png"
        img_path = os.path.join(images_dir, image_filename)
        img.save(img_path)

        
        annotation = {'boxes': boxes, 'labels': labels}
        annotation_filename = f"img_{img_idx:05d}.json"
        annotation_path = os.path.join(annotations_dir, annotation_filename)
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f)

class ShapesDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_filename)
        image = Image.open(image_path).convert('RGB')

        
        annotation_filename = image_filename.replace('.png', '.json')
        annotation_path = os.path.join(self.annotations_dir, annotation_filename)
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        boxes = annotation['boxes']
        labels = annotation['labels']

        if self.transform:
            image = self.transform(image)

        
        grid = torch.zeros((GRID_SIZE, GRID_SIZE, ANCHORS_PER_CELL * (5 + NUM_CLASSES)))

        cell_size = IMAGE_SIZE / GRID_SIZE

        for box, label in zip(boxes, labels):
            x_min, y_min, x_max, y_max = box
            box_width = x_max - x_min
            box_height = y_max - y_min
            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0

            
            grid_x = int(x_center / cell_size)
            grid_y = int(y_center / cell_size)
            if grid_x >= GRID_SIZE:
                grid_x = GRID_SIZE - 1
            if grid_y >= GRID_SIZE:
                grid_y = GRID_SIZE - 1

            
            x_rel = (x_center - grid_x * cell_size) / cell_size
            y_rel = (y_center - grid_y * cell_size) / cell_size
            w_rel = box_width / IMAGE_SIZE
            h_rel = box_height / IMAGE_SIZE

            
            assigned = False
            for anchor in range(ANCHORS_PER_CELL):
                if grid[grid_y, grid_x, anchor * (5 + NUM_CLASSES)] == 0:
                    grid[grid_y, grid_x, anchor * (5 + NUM_CLASSES) + 0] = 1  
                    grid[grid_y, grid_x, anchor * (5 + NUM_CLASSES) + 1] = x_rel
                    grid[grid_y, grid_x, anchor * (5 + NUM_CLASSES) + 2] = y_rel
                    grid[grid_y, grid_x, anchor * (5 + NUM_CLASSES) + 3] = w_rel
                    grid[grid_y, grid_x, anchor * (5 + NUM_CLASSES) + 4] = h_rel
                    grid[grid_y, grid_x, anchor * (5 + NUM_CLASSES) + 5 + label] = 1  
                    assigned = True
                    break
            if not assigned:
                
                pass  

        return image, grid


transform = transforms.Compose([
    transforms.ToTensor(),
])


class ShapeDetector(nn.Module):
    def __init__(self, grid_size=GRID_SIZE, num_classes=NUM_CLASSES, anchors_per_cell=ANCHORS_PER_CELL):
        super(ShapeDetector, self).__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.anchors_per_cell = anchors_per_cell
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),  
        )
        self.regressor = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, grid_size * grid_size * anchors_per_cell * (5 + num_classes)),
            
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        x = x.view(batch_size, -1)
        x = self.regressor(x)
        x = x.view(batch_size, self.grid_size, self.grid_size, self.anchors_per_cell * (5 + self.num_classes))
        return x

def compute_iou(pred_boxes, target_boxes):
    """Compute Intersection over Union (IoU) between two sets of boxes.

    Args:
        pred_boxes (Tensor): Predicted boxes of shape (..., 4) with [x_min, y_min, x_max, y_max]
        target_boxes (Tensor): Target boxes of shape (..., 4) with [x_min, y_min, x_max, y_max]

    Returns:
        Tensor: IoU values of shape (...)
    """
    
    inter_x_min = torch.max(pred_boxes[..., 0], target_boxes[..., 0])
    inter_y_min = torch.max(pred_boxes[..., 1], target_boxes[..., 1])
    inter_x_max = torch.min(pred_boxes[..., 2], target_boxes[..., 2])
    inter_y_max = torch.min(pred_boxes[..., 3], target_boxes[..., 3])

    
    inter_area = (inter_x_max - inter_x_min).clamp(min=0) * (inter_y_max - inter_y_min).clamp(min=0)

    
    pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]).clamp(min=0) * (pred_boxes[..., 3] - pred_boxes[..., 1]).clamp(min=0)
    target_area = (target_boxes[..., 2] - target_boxes[..., 0]).clamp(min=0) * (target_boxes[..., 3] - target_boxes[..., 1]).clamp(min=0)

    
    iou = inter_area / (pred_area + target_area - inter_area + 1e-6)
    return iou

def visualize_prediction(image, target, prediction, threshold=0.5, iou_threshold=0.3):
    """Visualize ground truth and predictions on the image with NMS applied.

    Args:
        image (Tensor): Image tensor of shape (3, H, W).
        target (dict): Dictionary containing 'boxes' and 'labels'.
        prediction (Tensor): Model predictions of shape (grid_size, grid_size, anchors_per_cell * (5 + num_classes)).
        threshold (float): Confidence threshold to filter predictions.
        iou_threshold (float): IoU threshold for NMS.
    """
    image = image.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(6,6))
    plt.imshow(image)
    ax = plt.gca()

    cell_size = IMAGE_SIZE / GRID_SIZE

    
    boxes_gt = []
    labels_gt = []
    boxes = target['boxes']
    labels = target['labels']
    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box
        boxes_gt.append([x_min, y_min, x_max, y_max])
        labels_gt.append(label)
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                             linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
        plt.text(x_min, y_min, CLASS_NAMES[label], color='green', fontsize=12)

    
    preds = prediction.cpu().detach().numpy()
    boxes_pred = []
    scores_pred = []
    labels_pred = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            for anchor in range(ANCHORS_PER_CELL):
                base_idx = anchor * (5 + NUM_CLASSES)
                conf = preds[i, j, base_idx + 0]
                if conf > threshold:
                    x_rel, y_rel, w_rel, h_rel = preds[i, j, base_idx + 1:base_idx + 5]
                    class_probs = preds[i, j, base_idx + 5:]
                    class_id = np.argmax(class_probs)
                    class_score = class_probs[class_id]

                    
                    x_center = (j + x_rel) * cell_size
                    y_center = (i + y_rel) * cell_size
                    box_width = w_rel * IMAGE_SIZE
                    box_height = h_rel * IMAGE_SIZE
                    x_min = x_center - box_width / 2
                    y_min = y_center - box_height / 2
                    x_max = x_center + box_width / 2
                    y_max = y_center + box_height / 2

                    boxes_pred.append([x_min, y_min, x_max, y_max])
                    scores_pred.append(conf)
                    labels_pred.append(class_id)

    if boxes_pred:
        
        boxes_pred_tensor = torch.tensor(boxes_pred).float().to(device)
        scores_pred_tensor = torch.tensor(scores_pred).to(device)
        labels_pred_tensor = torch.tensor(labels_pred).to(device)

        
        unique_labels = labels_pred_tensor.unique()
        keep_boxes = []
        keep_scores = []
        keep_labels = []
        for cls in unique_labels:
            cls_mask = labels_pred_tensor == cls
            cls_boxes = boxes_pred_tensor[cls_mask]
            cls_scores = scores_pred_tensor[cls_mask]
            if cls_boxes.numel() == 0:
                continue
            keep = nms(cls_boxes, cls_scores, iou_threshold)
            keep_boxes.append(cls_boxes[keep])
            keep_scores.append(cls_scores[keep])
            keep_labels.append(labels_pred_tensor[cls_mask][keep])

        if keep_boxes:
            boxes_nms = torch.cat(keep_boxes).cpu().numpy()
            scores_nms = torch.cat(keep_scores).cpu().numpy()
            labels_nms = torch.cat(keep_labels).cpu().numpy()

            for box, score, label in zip(boxes_nms, scores_nms, labels_nms):
                x_min, y_min, x_max, y_max = box
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                     linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                plt.text(x_min, y_min, f"{CLASS_NAMES[label]}: {score:.2f}", color='red', fontsize=12)

    plt.axis('off')
    plt.show()

def main():
    
    create_directories()

    
    print("Generating and saving training data...")
    generate_and_save_synthetic_data(NUM_TRAIN_IMAGES, TRAIN_IMAGES_DIR, TRAIN_ANNOTATIONS_DIR)
    print("Generating and saving validation data...")
    generate_and_save_synthetic_data(NUM_VAL_IMAGES, VAL_IMAGES_DIR, VAL_ANNOTATIONS_DIR)

    
    train_dataset = ShapesDataset(TRAIN_IMAGES_DIR, TRAIN_ANNOTATIONS_DIR, transform=transform)
    val_dataset = ShapesDataset(VAL_IMAGES_DIR, VAL_ANNOTATIONS_DIR, transform=transforms.ToTensor())  

    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    
    model = ShapeDetector().to(device)
    criterion_conf = nn.BCEWithLogitsLoss()
    criterion_class = nn.CrossEntropyLoss()
    criterion_coord = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)  

    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        loss_conf_total_epoch = 0.0
        loss_coord_total_epoch = 0.0
        loss_class_total_epoch = 0.0
        loss_iou_total_epoch = 0.0  

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            optimizer.zero_grad()
            outputs = model(images)

            
            outputs = outputs.view(batch_size, GRID_SIZE, GRID_SIZE, ANCHORS_PER_CELL, 5 + NUM_CLASSES)
            targets = targets.view(batch_size, GRID_SIZE, GRID_SIZE, ANCHORS_PER_CELL, 5 + NUM_CLASSES)

            
            conf = outputs[..., 0]
            target_conf = targets[..., 0]

            loss_conf = criterion_conf(conf, target_conf)

            
            coord_mask = target_conf > 0
            if coord_mask.sum() > 0:
                
                batch_idx, grid_y, grid_x, anchor_idx = torch.where(coord_mask)

                
                pred_boxes = outputs[batch_idx, grid_y, grid_x, anchor_idx, 1:5]
                target_boxes = targets[batch_idx, grid_y, grid_x, anchor_idx, 1:5]

                
                cell_size = IMAGE_SIZE / GRID_SIZE

                x_center_pred = (grid_x.float() + pred_boxes[:, 0]) * cell_size
                y_center_pred = (grid_y.float() + pred_boxes[:, 1]) * cell_size
                w_pred = pred_boxes[:, 2] * IMAGE_SIZE
                h_pred = pred_boxes[:, 3] * IMAGE_SIZE

                x_min_pred = x_center_pred - w_pred / 2
                y_min_pred = y_center_pred - h_pred / 2
                x_max_pred = x_center_pred + w_pred / 2
                y_max_pred = y_center_pred + h_pred / 2

                pred_boxes_abs = torch.stack([x_min_pred, y_min_pred, x_max_pred, y_max_pred], dim=1)

                x_center_target = (grid_x.float() + target_boxes[:, 0]) * cell_size
                y_center_target = (grid_y.float() + target_boxes[:, 1]) * cell_size
                w_target = target_boxes[:, 2] * IMAGE_SIZE
                h_target = target_boxes[:, 3] * IMAGE_SIZE

                x_min_target = x_center_target - w_target / 2
                y_min_target = y_center_target - h_target / 2
                x_max_target = x_center_target + w_target / 2
                y_max_target = y_center_target + h_target / 2

                target_boxes_abs = torch.stack([x_min_target, y_min_target, x_max_target, y_max_target], dim=1)

                
                iou = compute_iou(pred_boxes_abs, target_boxes_abs)
                loss_iou = (1 - iou).mean()
            else:
                loss_coord = torch.tensor(0.0).to(device)
                loss_iou = torch.tensor(0.0).to(device)

            
            class_mask = target_conf > 0
            if class_mask.sum() > 0:
                target_classes = torch.argmax(targets[..., 5:][class_mask], dim=-1)
                loss_class = criterion_class(outputs[..., 5:][class_mask], target_classes)
            else:
                loss_class = torch.tensor(0.0).to(device)

            
            if coord_mask.sum() > 0:
                loss_coord = criterion_coord(outputs[..., 1:5][coord_mask], targets[..., 1:5][coord_mask])
            else:
                loss_coord = torch.tensor(0.0).to(device)

            
            loss = loss_conf + loss_coord + loss_class + loss_iou
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss_conf_total_epoch += loss_conf.item()
            loss_coord_total_epoch += loss_coord.item()
            loss_class_total_epoch += loss_class.item()
            loss_iou_total_epoch += loss_iou.item()

        avg_loss = running_loss / len(train_loader)
        avg_conf_loss = loss_conf_total_epoch / len(train_loader)
        avg_coord_loss = loss_coord_total_epoch / len(train_loader)
        avg_class_loss = loss_class_total_epoch / len(train_loader)
        avg_iou_loss = loss_iou_total_epoch / len(train_loader)

        
        model.eval()
        val_loss = 0.0
        val_conf_loss = 0.0
        val_coord_loss = 0.0
        val_class_loss = 0.0
        val_iou_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                batch_size = images.size(0)
                outputs = model(images)
                
                outputs = outputs.view(batch_size, GRID_SIZE, GRID_SIZE, ANCHORS_PER_CELL, 5 + NUM_CLASSES)
                targets = targets.view(batch_size, GRID_SIZE, GRID_SIZE, ANCHORS_PER_CELL, 5 + NUM_CLASSES)
                
                conf = outputs[..., 0]
                target_conf = targets[..., 0]

                loss_conf = criterion_conf(conf, target_conf)

                coord_mask = target_conf > 0
                if coord_mask.sum() > 0:
                    batch_idx, grid_y, grid_x, anchor_idx = torch.where(coord_mask)
                    pred_boxes = outputs[batch_idx, grid_y, grid_x, anchor_idx, 1:5]
                    target_boxes = targets[batch_idx, grid_y, grid_x, anchor_idx, 1:5]
                    
                    cell_size = IMAGE_SIZE / GRID_SIZE

                    x_center_pred = (grid_x.float() + pred_boxes[:, 0]) * cell_size
                    y_center_pred = (grid_y.float() + pred_boxes[:, 1]) * cell_size
                    w_pred = pred_boxes[:, 2] * IMAGE_SIZE
                    h_pred = pred_boxes[:, 3] * IMAGE_SIZE

                    x_min_pred = x_center_pred - w_pred / 2
                    y_min_pred = y_center_pred - h_pred / 2
                    x_max_pred = x_center_pred + w_pred / 2
                    y_max_pred = y_center_pred + h_pred / 2

                    pred_boxes_abs = torch.stack([x_min_pred, y_min_pred, x_max_pred, y_max_pred], dim=1)

                    x_center_target = (grid_x.float() + target_boxes[:, 0]) * cell_size
                    y_center_target = (grid_y.float() + target_boxes[:, 1]) * cell_size
                    w_target = target_boxes[:, 2] * IMAGE_SIZE
                    h_target = target_boxes[:, 3] * IMAGE_SIZE

                    x_min_target = x_center_target - w_target / 2
                    y_min_target = y_center_target - h_target / 2
                    x_max_target = x_center_target + w_target / 2
                    y_max_target = y_center_target + h_target / 2

                    target_boxes_abs = torch.stack([x_min_target, y_min_target, x_max_target, y_max_target], dim=1)

                    
                    iou = compute_iou(pred_boxes_abs, target_boxes_abs)
                    loss_iou = (1 - iou).mean()
                else:
                    loss_coord = torch.tensor(0.0).to(device)
                    loss_iou = torch.tensor(0.0).to(device)

                
                class_mask = target_conf > 0
                if class_mask.sum() > 0:
                    target_classes = torch.argmax(targets[..., 5:][class_mask], dim=-1)
                    loss_class = criterion_class(outputs[..., 5:][class_mask], target_classes)
                else:
                    loss_class = torch.tensor(0.0).to(device)

                
                if coord_mask.sum() > 0:
                    loss_coord = criterion_coord(outputs[..., 1:5][coord_mask], targets[..., 1:5][coord_mask])
                else:
                    loss_coord = torch.tensor(0.0).to(device)

                
                loss = loss_conf + loss_coord + loss_class + loss_iou
                val_loss += loss.item()
                val_conf_loss += loss_conf.item()
                val_coord_loss += loss_coord.item()
                val_class_loss += loss_class.item()
                val_iou_loss += loss_iou.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_conf_loss = val_conf_loss / len(val_loader)
        avg_val_coord_loss = val_coord_loss / len(val_loader)
        avg_val_class_loss = val_class_loss / len(val_loader)
        avg_val_iou_loss = val_iou_loss / len(val_loader)

        
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}], "
              f"Training Loss: {avg_loss:.4f} "
              f"(Conf: {avg_conf_loss:.4f}, Coord: {avg_coord_loss:.4f}, Class: {avg_class_loss:.4f}, IoU: {avg_iou_loss:.4f}), "
              f"Validation Loss: {avg_val_loss:.4f} "
              f"(Conf: {avg_val_conf_loss:.4f}, Coord: {avg_val_coord_loss:.4f}, Class: {avg_val_class_loss:.4f}, IoU: {avg_val_iou_loss:.4f})")

    print("Training completed.")

    
    
    sample_idx = random.randint(0, len(val_dataset) - 1)
    sample_img, _ = val_dataset[sample_idx]
    sample_img_tensor = sample_img.unsqueeze(0).to(device)
    sample_annotation_path = os.path.join(VAL_ANNOTATIONS_DIR, val_dataset.image_files[sample_idx].replace('.png', '.json'))
    with open(sample_annotation_path, 'r') as f:
        sample_target = json.load(f)

    model.eval()
    with torch.no_grad():
        sample_pred = model(sample_img_tensor)
    sample_pred = sample_pred.squeeze(0)

    
    visualize_prediction(sample_img, sample_target, sample_pred, threshold=0.5, iou_threshold=0.3)

if __name__ == '__main__':
    main()
