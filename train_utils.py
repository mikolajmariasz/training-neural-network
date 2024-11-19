import torch
import torch.nn as nn

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_bbox = nn.SmoothL1Loss()
    running_loss = 0.0
    for i, (images, targets) in enumerate(data_loader):
        images = torch.stack(images).to(device)
        optimizer.zero_grad()
        pred_bbox, pred_cls = model(images)
        loss_cls = 0.0
        loss_bbox = 0.0
        total_objects = 0
        for j in range(images.size(0)):
            num_objs = targets[j]["boxes"].shape[0]
            if num_objs == 0:
                labels = torch.tensor([0], dtype=torch.long).to(device)
                pred_cls_j = pred_cls[j].unsqueeze(0)
                loss_cls += criterion_cls(pred_cls_j, labels)
                continue
            total_objects += num_objs
            for k in range(num_objs):
                labels = targets[j]["labels"][k].unsqueeze(0).to(device)
                bbox = targets[j]["boxes"][k].unsqueeze(0).to(device)
                pred_cls_j = pred_cls[j].unsqueeze(0)
                pred_bbox_j = pred_bbox[j].unsqueeze(0)
                loss_cls += criterion_cls(pred_cls_j, labels)
                loss_bbox += criterion_bbox(pred_bbox_j, bbox)
        if total_objects > 0:
            loss_cls /= total_objects
            loss_bbox /= total_objects
        else:
            loss_cls /= images.size(0)
            loss_bbox /= images.size(0)
        loss = loss_cls + loss_bbox
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % print_freq == 0:
            print(f"Epoch [{epoch}], Step [{i}/{len(data_loader)}], Loss: {loss.item():.4f}")
