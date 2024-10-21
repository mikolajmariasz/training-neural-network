import torch  # Główna biblioteka PyTorch do obliczeń numerycznych i budowy sieci neuronowych
import torch.nn as nn  # Moduł do tworzenia warstw sieci neuronowych
import torch.optim as optim  # Moduł do optymalizacji, np. SGD, Adam itp.
import torchvision  # Biblioteka do przetwarzania obrazów, zawiera popularne zestawy danych i narzędzia do przetwarzania obrazów
import torchvision.transforms as transforms  # Moduł do przekształcania danych, np. normalizacja, konwersja na tensor itp.
import matplotlib.pyplot as plt  # Biblioteka do tworzenia wykresów i wizualizacji
import numpy as np  # Biblioteka numeryczna do operacji na macierzach/tablicach
from torch.utils.data import DataLoader, Subset  # DataLoader do ładowania danych w mini-batchach, Subset do filtrowania zbioru danych

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
    for idx in range(len(dataset)):
        # Przemapowujemy oryginalną etykietę na nowy indeks zgodny z wybranymi klasami
        dataset.dataset.targets[dataset.indices[idx]] = selected_indices.index(dataset.dataset.targets[dataset.indices[idx]])

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