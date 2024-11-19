import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

def visualize_predictions(model, data_loader, device, num_images=5):
    model.eval()
    with torch.no_grad():
        count = 0
        for images, targets in data_loader:
            images = torch.stack(images).to(device)
            pred_bbox, pred_cls = model(images)
            for j in range(images.size(0)):
                if count >= num_images:
                    return
                img = images[j].cpu().permute(1, 2, 0).numpy()
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                for box in targets[j]["boxes"]:
                    xmin = int(box[0] * 256)
                    ymin = int(box[1] * 256)
                    xmax = int(box[2] * 256)
                    ymax = int(box[3] * 256)
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                pred_box = pred_bbox[j].cpu().numpy()
                pred_label = torch.argmax(pred_cls[j]).item()
                if pred_label == 1:
                    xmin = int(pred_box[0] * 256)
                    ymin = int(pred_box[1] * 256)
                    xmax = int(pred_box[2] * 256)
                    ymax = int(pred_box[3] * 256)
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                plt.figure(figsize=(8, 8))
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
                count += 1
                if count >= num_images:
                    return
