import torch  # Główna biblioteka PyTorch do obliczeń numerycznych i budowy sieci neuronowych
import torch.nn as nn  # Moduł do tworzenia warstw sieci neuronowych
import torch.optim as optim  # Moduł do optymalizacji, np. SGD, Adam itp.
import torchvision  # Biblioteka do przetwarzania obrazów, zawiera popularne zestawy danych i narzędzia do przetwarzania obrazów
import torchvision.transforms as transforms  # Moduł do przekształcania danych, np. normalizacja, konwersja na tensor itp.
from torch.utils.data import DataLoader, Subset  # DataLoader do ładowania danych w mini-batchach, Subset do filtrowania zbioru danych
import matplotlib.pyplot as plt  # Biblioteka do tworzenia wykresów i wizualizacji
import numpy as np  # Biblioteka numeryczna do operacji na macierzach/tablicach

from model import EnhancedCNN # importowanie struktury modelu sieci neuronowej


# Transformacja obrazu do tensora PyTorch i normalizacja wartości pikseli do zakresu [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),  # Konwertuje obraz do formatu tensora (z wartościami pikseli [0, 255] na [0, 1])
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizuje każdy kanał (R, G, B) na zakres [-1, 1]
])

# Załadowanie zbioru treningowego CIFAR-10
# CIFAR-10 to zbiór danych obrazów 32x32 piksele z 10 różnymi klasami (samoloty, samochody, ptaki itp.)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Załadowanie zbioru testowego CIFAR-10
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Klasy w zbiorze danych CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Wybrane klasy, które chcemy wykrywać w naszym modelu
selected_classes = ['plane', 'car', 'bird', 'cat', 'dog', 'truck']

# Pobieranie indeksów klas dla wybranych etykiet
# Zamieniamy nazwy klas na ich indeksy, np. 'plane' -> 0, 'car' -> 1 itd.
selected_class_indices = [classes.index(cls) for cls in selected_classes]

# Filtrowanie zbioru treningowego
# Sprawdzamy, które obrazy w zbiorze treningowym należą do wybranych klas, i zapisujemy ich indeksy
train_indices = [i for i, label in enumerate(trainset.targets) if label in selected_class_indices]

# Tworzymy nowy zestaw danych, który zawiera tylko wybrane klasy, wykorzystując Subset
filtered_trainset = Subset(trainset, train_indices)

# Filtrowanie zbioru testowego
# Analogicznie do zbioru treningowego, filtrujemy również zbiór testowy
test_indices = [i for i, label in enumerate(testset.targets) if label in selected_class_indices]

# Tworzymy nowy zestaw testowy z przefiltrowanymi klasami
filtered_testset = Subset(testset, test_indices)

# Funkcja do przemapowania etykiet
# Zamieniamy oryginalne etykiety na nowe indeksy (0, 1, 2, ...) dla wybranych klas
def remap_labels(dataset, selected_indices):
    new_labels = []
    for idx in range(len(dataset)):
        new_labels.append(selected_indices.index(dataset.dataset.targets[dataset.indices[idx]]))
    dataset.dataset.targets = new_labels

# Przemapowanie etykiet w zbiorze treningowym i testowym
remap_labels(filtered_trainset, selected_class_indices)
remap_labels(filtered_testset, selected_class_indices)

# Oblicz rozmiar zestawu treningowego (80%) i walidacyjnego (20%)
train_size = int(0.8 * len(filtered_trainset))  # 80% na trening
val_size = len(filtered_trainset) - train_size  # 20% na walidację

# Podział na zestaw treningowy i walidacyjny
train_dataset, val_dataset = torch.utils.data.random_split(filtered_trainset, [train_size, val_size])

# Tworzenie DataLoaderów dla zbioru treningowego, walidacyjnego i testowego
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)  # Mini-batche po 32 obrazy, dane są losowo mieszane
valloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)  # Mini-batche po 32 obrazy, dane nie są mieszane
testloader = DataLoader(filtered_testset, batch_size=32, shuffle=False, num_workers=2)  # Mini-batche po 32 obrazy, dane nie są mieszane



# Inicjalizacja sieci, funkcji kosztu oraz optymalizatora z nowym współczynnikiem uczenia
net = EnhancedCNN(num_classes=len(selected_classes))  # Tworzymy instancję sieci
criterion = nn.CrossEntropyLoss()  # Funkcja kosztu (CrossEntropyLoss dla klasyfikacji wieloklasowej)
optimizer = optim.Adam(net.parameters(), lr=0.0005)  # Optymalizator Adam z niskim współczynnikiem uczenia (stabilność)

def train_model():
    train_losses, val_losses = [], []
    for epoch in range(10):
        running_loss = 0.0
        net.train()
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(trainloader))

        # Ewaluacja na zbiorze walidacyjnym po każdej epoce
        val_loss = 0.0
        net.eval()
        with torch.no_grad():
            for inputs, labels in valloader:
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_losses.append(val_loss / len(valloader))
        
        print(f"Epoch {epoch+1} - Train Loss: {train_losses[-1]:.4f} - Validation Loss: {val_losses[-1]:.4f}")

    # Zapisz model
    torch.save(net.state_dict(), 'enhanced_cnn.pth')
    print("Model zapisany do pliku 'enhanced_cnn.pth'")

# Uruchomienie treningu
train_model()