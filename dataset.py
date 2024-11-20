# dataset.py

import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np

class PennFudanDataset(Dataset):
    """
    Dataset class for the Penn-Fudan Pedestrian Detection
    - Handles loading images and corresponding masks, proccessing them to bounding boxesa and labels
    """
    def __init__(self, root, transforms=None, target_size=(256, 256)):
        """
        Initialize dataset by listing all image and mask files

        Args:
            root (str): Root directory of the dataset
            transforms (): Optional transforms for future data augumentation
            target_size (tuple): Size to which all images will be resized
        """
        self.root = root
        self.transforms = transforms
        self.target_size = target_size
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages")))) # Sorts all images
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks")))) # Sorts all masks

    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding target (bounding boxes and labels) by index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: (image, target) where target is a dictionary containing:
                - 'boxes' (Tensor): Bounding boxes (normalized)
                - 'labels' (Tensor): Labels for each bounding box (1 for pedestrian, 0 for bg).
        """
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB") # Converts PNGs to RGB format
        mask = Image.open(mask_path)
        img = img.resize(self.target_size, resample=Image.BILINEAR) # Resize images (Bilinear)
        mask = mask.resize(self.target_size, resample=Image.NEAREST)  # Resize masks (NN)
        
        if self.transforms:
            img = self.transforms(img) # Apply transforms
        
        mask = np.array(mask)
        obj_ids = np.unique(mask)[1:]  # Exclude background, leave only object IDs
        
        boxes = [] # List storing bounding boxes
        labels = [] # List storing labels
        for obj_id in obj_ids:
            # Find position of mask that equals to current object ID
            pos = np.where(mask == obj_id)
            # Calculate bounding boxes
            if pos[0].size > 0 and pos[1].size > 0:
                xmin = np.min(pos[1]) / self.target_size[1]
                xmax = np.max(pos[1]) / self.target_size[1]
                ymin = np.min(pos[0]) / self.target_size[0]
                ymax = np.max(pos[0]) / self.target_size[0]
                boxes.append([xmin, ymin, xmax, ymax]) # Append list by bounding box cords of pedestrian
                labels.append(1) # Append list by 1 for pedestrian
        
        # If any bounding boxes were found, convert them to tensors
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        # Creates empty tensor for boxes and labels
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        # Target dictionary containing bboxes and labels
        target = {"boxes": boxes, "labels": labels}
        return img, target

    def __len__(self):
        """
        Returns:
            int: Number of images in the dataset
        """
        return len(self.imgs)
