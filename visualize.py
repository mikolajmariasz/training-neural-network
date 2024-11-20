# visualize.py

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torchvision.ops as ops

def compute_iou(box1, box2):
    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])

    inter_width = max(0, xmax - xmin)
    inter_height = max(0, ymax - ymin)
    inter_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    else:
        return inter_area / union_area

def visualize_predictions(model, data_loader, device, num_images=5, iou_threshold=0.5, score_threshold=0.5):
    """
    Function to evaluate and visualize predictions after model training

    Args:
        model (nn.Module): Model for evaluation.
        device (): Device for computation
        num_images (int): Number of images to show
        iou_threshold (float): IoU threshold for Non-Maximum Suppression (NMS)
        score_threshold (float): Score threshold to filter weak predictions
    """
    model.eval() # Evaluation mode
    with torch.no_grad(): # Disable gradient
        count = 0
        for images, targets in data_loader:
            images = torch.stack(images).to(device) # Move images to device
            pred_bboxes, pred_cls = model(images) # Obtain predictions
            # Show images
            for j in range(images.size(0)):
                if count >= num_images:
                    return
                img = images[j].cpu().permute(1, 2, 0).numpy() # Convert tensor to numpy
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]) # Unnormalize
                img = np.clip(img * 255, 0, 255).astype(np.uint8) # Scale and clip
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert to BGR

                # Draw ground truth boxes
                gt_boxes = targets[j]["boxes"].cpu().numpy()
                for gt_box in gt_boxes:
                    xmin = int(gt_box[0] * 256)
                    ymin = int(gt_box[1] * 256)
                    xmax = int(gt_box[2] * 256)
                    ymax = int(gt_box[3] * 256)
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Green for GT

                # Extract scores for class '1' (pedestrian)
                scores = pred_cls[j][:, 1] 

                # Apply NMS
                keep = ops.nms(pred_bboxes[j], scores, iou_threshold)
                keep = keep[scores[keep] > score_threshold]

                # Draw predicted boxes after NMS
                pred_boxes = pred_bboxes[j][keep].cpu().numpy()
                pred_labels = torch.argmax(pred_cls[j][keep], dim=1).cpu().numpy()
                for p in range(pred_boxes.shape[0]):
                    if pred_labels[p] == 0:
                        continue  # Skip background
                    pred_box = pred_boxes[p]
                    xmin = int(pred_box[0] * 256)
                    ymin = int(pred_box[1] * 256)
                    xmax = int(pred_box[2] * 256)
                    ymax = int(pred_box[3] * 256)
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Blue for Prediction

                plt.figure(figsize=(8, 8))
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title(f'Training Prediction {count+1}')
                plt.show()
                count += 1
                if count >= num_images:
                    return

def visualize_initial_predictions(model, data_loader, device, num_images=5):
    """
    Visualizes model predictions before training

    Args:
        model (torch.nn.Module): Model to evaluate
        data_loader (DataLoader): Dataloader providing images and ground truth targets
        device (torch.device): Device for computation
        num_images (int): Number of images to visualize
    """
    model.eval()
    with torch.no_grad():
        count = 0
        for images, targets in data_loader:
            images = torch.stack(images).to(device)
            pred_bboxes, pred_cls = model(images)
            for j in range(images.size(0)):
                if count >= num_images:
                    return
                img = images[j].cpu().permute(1, 2, 0).numpy()
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # Draw ground truth boxes
                gt_boxes = targets[j]["boxes"].cpu().numpy()
                for gt_box in gt_boxes:
                    xmin = int(gt_box[0] * 256)
                    ymin = int(gt_box[1] * 256)
                    xmax = int(gt_box[2] * 256)
                    ymax = int(gt_box[3] * 256)
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Green for GT

                # Draw predicted boxes
                pred_boxes = pred_bboxes[j].cpu().numpy()
                pred_labels = torch.argmax(pred_cls[j], dim=1).cpu().numpy()
                for p in range(pred_boxes.shape[0]):
                    if pred_labels[p] == 0:
                        continue  # Skip background
                    pred_box = pred_boxes[p]
                    xmin = int(pred_box[0] * 256)
                    ymin = int(pred_box[1] * 256)
                    xmax = int(pred_box[2] * 256)
                    ymax = int(pred_box[3] * 256)
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Blue for Prediction

                plt.figure(figsize=(8, 8))
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title(f'Initial Prediction {count+1}')
                plt.show()
                count += 1
                if count >= num_images:
                    return
