# train_utils.py

import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment

def collate_fn(batch):
    """
    Custom collate function to handle batches with varying number of objects.
    
    Args:
        batch (list): A list of tuples where each tuple contains an image and its target.
    
    Returns:
        tuple: A tuple containing a list of images and a list of targets.
    """
    return tuple(zip(*batch))

def compute_iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1 (list or np.array): Coordinates [xmin, ymin, xmax, ymax] of the first box.
        box2 (list or np.array): Coordinates [xmin, ymin, xmax, ymax] of the second box.
    
    Returns:
        float: The IoU between box1 and box2.
    """
    # cordinates of intersection rectangle
    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])

    # width and height of the intersection rectangle
    inter_width = max(0, xmax - xmin)
    inter_height = max(0, ymax - ymin)
    inter_area = inter_width * inter_height

    # area of bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    else:
        return inter_area / union_area # IoU

def assign_predictions_to_targets(pred_bboxes, pred_cls, targets, iou_threshold=0.2):  # Lowered threshold
    """
    Assigns model predictions to ground truth boxes using the Hungarian Algorithm based on IoU.
    
    Args:
        pred_bboxes (Tensor): Predicted bounding boxes from the model of shape (batch_size, num_predictions, 4).
        pred_cls (Tensor): Predicted class scores from the model of shape (batch_size, num_predictions, num_classes).
        targets (list): List of target dictionaries for each image containing 'boxes' and 'labels'.
        iou_threshold (float, optional): Minimum IoU required to consider a prediction as a match. Defaults to 0.2.
    
    Returns:
        tuple:
            - matched_labels (Tensor): Tensor of matched class labels of shape (batch_size, num_predictions).
            - target_bboxes (Tensor): Tensor of target bounding boxes of shape (batch_size, num_predictions, 4).
    """
    batch_size, num_predictions, _ = pred_bboxes.shape # Extract batch size and number of predictions
    matched_labels = []
    target_bboxes = []
    for b in range(batch_size):
        gt_boxes = targets[b]['boxes'].cpu().numpy() 
        gt_labels = targets[b]['labels'].cpu().numpy() 
        num_gt = gt_boxes.shape[0]
        num_pred = num_predictions # Number of predictions

        if num_gt == 0:
            # No objects in the image (all predictions to bg)
            matched_labels.append(torch.zeros(num_predictions, dtype=torch.long))
            target_bboxes.append(torch.zeros((num_predictions, 4), dtype=torch.float32))
            continue
        
        iou_matrix = np.zeros((num_pred, num_gt))
        for p in range(num_pred):
            for t in range(num_gt):
                # Detach the tensor before converting to numpy
                iou_matrix[p, t] = compute_iou(pred_bboxes[b, p].detach().cpu().numpy(), gt_boxes[t])

        # print(f"Image {b+1} IoU Matrix:\n{iou_matrix}")

        # Apply Hungarian Algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # Maximize IoU

        # Initialize labels as background
        target_labels_np = np.zeros(num_pred, dtype=np.int64)

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_threshold:
                target_labels_np[r] = gt_labels[c] # class label
            else:
                target_labels_np[r] = 0  # background label

        matched_labels.append(torch.tensor(target_labels_np, dtype=torch.long)) # Add matched labels to the list

        # Assign corresponding bounding boxes
        assigned_boxes = []
        for p in range(num_pred):
            if target_labels_np[p] == 0:
                assigned_boxes.append(torch.zeros(4))
            else:
                # Find the ground truth box assigned to this prediction
                assigned_gt = col_ind[np.where(row_ind == p)]
                if assigned_gt.size > 0:
                    assigned_boxes.append(torch.tensor(gt_boxes[assigned_gt[0]], dtype=torch.float32))
                else:
                    assigned_boxes.append(torch.zeros(4))
        assigned_boxes = torch.stack(assigned_boxes)
        target_bboxes.append(assigned_boxes)

        num_matches = (target_labels_np > 0).sum()
        print(f"Image {b+1}: {num_matches} predictions matched to ground truths.")

    matched_labels = torch.stack(matched_labels).to(pred_cls.device)  # (batch_size, num_predictions)
    target_bboxes = torch.stack(target_bboxes).to(pred_bboxes.device)  # (batch_size, num_predictions, 4)

    return matched_labels, target_bboxes

def iou_loss(pred_boxes, target_boxes):
    """
    Computes the IoU loss
    
    Args:
        pred_boxes (Tensor): Predicted bounding boxes of shape (N, 4)
        target_boxes (Tensor): Ground truth bounding boxes of shape (N, 4)
    
    Returns:
        Tensor: Scalar tensor representing the average IoU loss.
    """
    # cordinates of intersection rectangle
    inter_xmin = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_ymin = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_xmax = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_ymax = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    # width and height of the intersection rectangle
    inter_width = (inter_xmax - inter_xmin).clamp(min=0)
    inter_height = (inter_ymax - inter_ymin).clamp(min=0)
    inter_area = inter_width * inter_height

    # area of bounding boxes
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])

    union_area = pred_area + target_area - inter_area + 1e-6  # Prevent division by zero

    iou = inter_area / union_area

    loss = 1 - iou  # IoU loss
    return loss.mean()

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """
    Trains the model for one epoch.
    
    Args:
        model (nn.Module): Model to train.
        optimizer (Optimizer): The optimizer to update model parameters.
        data_loader (DataLoader): DataLoader for the training dataset.
        device (torch.device): Device to perform computations on.
        epoch (int): Current epoch number.
        print_freq (int, optional): Frequency (in steps) to print training status. Defaults to 10.
    
    Returns:
        None
    """
    model.train() # training mode
    criterion_cls = nn.CrossEntropyLoss() # classification loss function
    running_loss = 0.0
    running_loss_cls = 0.0
    running_loss_bbox = 0.0
    total_iou = 0.0
    total_matches = 0

    for i, (images, targets) in enumerate(data_loader):
        images = torch.stack(images).to(device)
        optimizer.zero_grad()
        pred_bboxes, pred_cls = model(images)  # pred_bboxes: (batch, num_preds, 4), pred_cls: (batch, num_preds, num_classes)

        # Assign predictions to targets
        matched_labels, target_bboxes = assign_predictions_to_targets(pred_bboxes, pred_cls, targets, iou_threshold=0.2)

        # Classification loss
        loss_cls = criterion_cls(pred_cls.view(-1, model.num_classes), matched_labels.view(-1))

        # Bounding box loss using IoU Loss
        pos_indices = matched_labels > 0
        if pos_indices.sum() > 0:
            loss_bbox = iou_loss(pred_bboxes[pos_indices], target_bboxes[pos_indices])
        else:
            loss_bbox = torch.tensor(0.0, requires_grad=True).to(device)

        loss = loss_cls + loss_bbox # total loss
        loss.backward() # backpropagation
        optimizer.step() # Updates model parameters

        # Accumulate loss (only for reporting)
        running_loss += loss.item()
        running_loss_cls += loss_cls.item()
        running_loss_bbox += loss_bbox.item()

        # IoU for metrics
        with torch.no_grad():
            for b in range(images.size(0)):
                gt_boxes = targets[b]['boxes'].cpu().numpy() # Ground truth boxes for the b-th image
                pred_boxes = pred_bboxes[b].cpu().numpy() # Predicted boxes for the b-th image
                for gt_box in gt_boxes: 
                    # Compute IoUs between the gt_box and all predicted boxes
                    ious = [compute_iou(gt_box, pred_box) for pred_box in pred_boxes]
                    max_iou = max(ious) if ious else 0.0 # minimum IoU
                    total_iou += max_iou # Accumulate IoU
                    total_matches += 1

        if i % print_freq == 0:
            # Calculate average losses and IoU up to the current step
            avg_loss = running_loss / (i + 1)
            avg_loss_cls = running_loss_cls / (i + 1)
            avg_loss_bbox = running_loss_bbox / (i + 1)
            avg_iou = total_iou / total_matches if total_matches > 0 else 0
            print(f"Epoch [{epoch+1}], Step [{i+1}/{len(data_loader)}], "
                  f"Loss: {avg_loss:.4f} (Cls: {avg_loss_cls:.4f}, BBox: {avg_loss_bbox:.4f}), "
                  f"Avg IoU: {avg_iou:.4f}")
            
    # Calculate average losses and IoU for epoch
    epoch_loss = running_loss / len(data_loader)
    epoch_loss_cls = running_loss_cls / len(data_loader)
    epoch_loss_bbox = running_loss_bbox / len(data_loader)
    epoch_iou = total_iou / total_matches if total_matches > 0 else 0
    print(f"Epoch [{epoch+1}] completed. Avg Loss: {epoch_loss:.4f} "
          f"(Cls: {epoch_loss_cls:.4f}, BBox: {epoch_loss_bbox:.4f}), Avg IoU: {epoch_iou:.4f}")
