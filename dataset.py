import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np

class PennFudanDataset(Dataset):
    def __init__(self, root, transforms=None, target_size=(256, 256)):
        self.root = root
        self.transforms = transforms
        self.target_size = target_size
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        img = img.resize(self.target_size, resample=Image.BILINEAR)
        mask = mask.resize(self.target_size, resample=Image.NEAREST)
        if self.transforms:
            img = self.transforms(img)
        mask = np.array(mask)
        obj_ids = np.unique(mask)[1:]
        masks = mask == obj_ids[:, None, None]
        boxes = []
        for i in range(len(obj_ids)):
            pos = np.where(masks[i])
            if pos[0].size > 0 and pos[1].size > 0:
                xmin = np.min(pos[1]) / self.target_size[1]
                xmax = np.max(pos[1]) / self.target_size[1]
                ymin = np.min(pos[0]) / self.target_size[0]
                ymax = np.max(pos[0]) / self.target_size[0]
                boxes.append([xmin, ymin, xmax, ymax])
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}
        return img, target

    def __len__(self):
        return len(self.imgs)
