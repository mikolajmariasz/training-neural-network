import torch.nn as nn

class SimpleObjectDetector(nn.Module):
    def __init__(self, num_classes):
        super(SimpleObjectDetector, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.flatten = nn.Flatten()
        self.flat_size = 64 * 32 * 32
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.ReLU(),
        )
        self.bbox_head = nn.Linear(512, 4)
        self.cls_head = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        bbox = self.bbox_head(x)
        cls_logits = self.cls_head(x)
        return bbox, cls_logits
