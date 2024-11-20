# model.py

import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleObjectDetector(nn.Module):
    """
    A simple CNN for object detection
    The model predicts bounding boxes and class scores for images
    """
    def __init__(self, num_classes, num_predictions=5):
        """
        Initializes the model's layers and parameters

        Args:
            num_classes (int): Number of object classes to predict
            num_predictions (int, optional): Number of bounding box predictions per image
        """
        super(SimpleObjectDetector, self).__init__()
        self.num_predictions = num_predictions
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  #  Conv layer input channels=3 (RGB)
            nn.ReLU(), # Activation function
            nn.MaxPool2d(2, 2),  # Max pooling with kernel size=2 and stride=2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # Conv layer
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),  # 64x64
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Increased channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32
        )
        self.flatten = nn.Flatten() # Flatens 2D map to vector
        self.flat_size = 128 * 32 * 32  # Calculate size after flattening
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, 1024), # input = flat_size, output = 1024
            nn.ReLU(),
        )
        # Output layers for bboxes and classes
        self.bbox_head = nn.Linear(1024, self.num_predictions * 4)  # 4 coordinates per prediction
        self.cls_head = nn.Linear(1024, self.num_predictions * self.num_classes)  # Class scores per prediction

        # Initialize bbox_head biases to spread initial predictions
        self.initialize_bbox_head()

    def initialize_bbox_head(self):
        """
        Initializes the biases of the bbox_head to predefined bounding boxes spread across the image
        This helps the model start with predictions in different positions
        """
        # Define initial bounding boxes spread across the image
        # num_predictions=5
        initial_boxes = [
            [0.2, 0.2, 0.4, 0.4],
            [0.6, 0.2, 0.8, 0.4],
            [0.2, 0.6, 0.4, 0.8],
            [0.6, 0.6, 0.8, 0.8],
            [0.4, 0.4, 0.6, 0.6],
        ]
        # Flatten the list and convert to tensor
        initial_boxes = [coord for box in initial_boxes for coord in box]
        initial_boxes = torch.tensor(initial_boxes, dtype=torch.float32)
        # Apply sigmoid since bbox coordinates are between 0 and 1
        initial_boxes = torch.sigmoid(initial_boxes)
        # Set the bias of bbox_head
        with torch.no_grad():
            self.bbox_head.bias.copy_(initial_boxes)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (Tensor): Input image tensor of shape (batch_size, 3 (rgb), height, width).

        Returns:
            tuple:
                - bbox (Tensor): Predicted bounding boxes of shape (batch_size, num_predictions, 4).
                - cls_logits (Tensor): Predicted class score of shape (batch_size, num_predictions, num_classes).
        """
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        bbox = self.bbox_head(x)  # Shape: (batch_size, num_predictions * 4)
        cls_logits = self.cls_head(x)  # Shape: (batch_size, num_predictions * num_classes)

        # Apply sigmoid activation (bbox coordinates between 0 and 1)
        bbox = torch.sigmoid(bbox)

        # Reshape bbox and cls_logits to separate predictions
        bbox = bbox.view(-1, self.num_predictions, 4)
        cls_logits = cls_logits.view(-1, self.num_predictions, self.num_classes)

        return bbox, cls_logits
