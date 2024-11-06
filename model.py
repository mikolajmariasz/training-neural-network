import torch
import torch.nn as nn

class EnhancedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(EnhancedCNN, self).__init__()
        # Warstwy konwolucyjne z batch normalization i dropout
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 32 filtry w warstwie pierwszej
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization dla pierwszej warstwy konwolucyjnej
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 64 filtry w warstwie drugiej
        self.bn2 = nn.BatchNorm2d(64)  # Batch Normalization dla drugiej warstwy konwolucyjnej
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # 128 filtrów w warstwie trzeciej
        self.bn3 = nn.BatchNorm2d(128)  # Batch Normalization dla trzeciej warstwy konwolucyjnej
        self.pool = nn.MaxPool2d(2, 2)  # Max Pooling z oknem 2x2

        # Warstwa w pełni połączona
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # 256 neuronów w warstwie w pełni połączonej
        self.dropout = nn.Dropout(0.5)  # Dropout dla regularizacji (50% prawdopodobieństwo wyzerowania neuronu)
        self.fc2 = nn.Linear(256, num_classes)  # Wyjście odpowiada liczbie klas

    def forward(self, x):
        # Forward pass z funkcją ReLU, poolingiem i batch normalization
        # Przepuszczamy dane przez pierwszą warstwę konwolucyjną, funkcję ReLU i Batch Normalization, potem Max Pooling
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        # Analogicznie dla drugiej warstwy
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        # Analogicznie dla trzeciej warstwy
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        # Flattening - zmiana rozmiaru obrazu do wektora, aby przejść do warstwy w pełni połączonej
        x = x.view(-1, 128 * 4 * 4)
        # Warstwa w pełni połączona z funkcją ReLU
        x = torch.relu(self.fc1(x))
        # Dropout przed ostatnią warstwą
        x = self.dropout(x)  
        # Ostateczna warstwa wyjściowa
        x = self.fc2(x)
        return x
