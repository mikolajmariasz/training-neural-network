import torch
from model import EnhancedCNN
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Załadowanie danych
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# Ładowanie modelu
net = EnhancedCNN()
net.load_state_dict(torch.load('enhanced_cnn.pth'))
net.eval()  # Przełączamy model w tryb ewaluacji

# Funkcja ewaluacji modelu
def evaluate_model():
    correct = 0
    total = 0
    with torch.no_grad():  # Wyłączamy obliczanie gradientów
        for inputs, labels in testloader:
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")

# Testowanie modelu
evaluate_model()