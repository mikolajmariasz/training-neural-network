import torch  # Główna biblioteka PyTorch do obliczeń numerycznych i budowy sieci neuronowych
import torch.nn as nn  # Moduł do tworzenia warstw sieci neuronowych
import torch.optim as optim  # Moduł do optymalizacji, np. SGD, Adam itp.
import torchvision  # Biblioteka do przetwarzania obrazów, zawiera popularne zestawy danych i narzędzia do przetwarzania obrazów
import torchvision.transforms as transforms  # Moduł do przekształcania danych, np. normalizacja, konwersja na tensor itp.
from torch.utils.data import DataLoader  # DataLoader do ładowania danych w mini-batchach
import matplotlib.pyplot as plt  # Biblioteka do tworzenia wykresów i wizualizacji
import numpy as np  # Biblioteka numeryczna do operacji na macierzach/tablicach

from model import EnhancedCNN  # Importowanie struktury modelu sieci neuronowej

def main():
    # Transformacja obrazu do tensora PyTorch i normalizacja wartości pikseli do zakresu [-1, 1]
    transform = transforms.Compose([ 
        transforms.ToTensor(),  # Konwertuje obraz do formatu tensora (z wartościami pikseli [0, 255] na [0, 1])
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizuje każdy kanał (R, G, B) na zakres [-1, 1]
    ])

    # Załadowanie zbioru treningowego CIFAR-10
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    print(f"Rozmiar zbioru danych treningowych: {len(trainset)}")  # Komunikat o rozmiarze zbioru danych

    # Wizualizacja przykładowego obrazu (tylko raz)
    img, label = trainset[0]
    plt.imshow(np.transpose(img, (1, 2, 0)))  # Przekształcamy kanały obrazu, aby pasowały do formatu RGB
    plt.title(f'Etykieta: {label}')
    plt.show()

    # Załadowanie zbioru testowego CIFAR-10
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Klasy w zbiorze danych CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Oblicz rozmiar zestawu treningowego (80%) i walidacyjnego (20%)
    train_size = int(0.8 * len(trainset))  # 80% na trening
    val_size = len(trainset) - train_size  # 20% na walidację

    # Podział na zestaw treningowy i walidacyjny
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])

    # Tworzenie DataLoaderów dla zbioru treningowego, walidacyjnego i testowego
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)  # Mini-batche po 32 obrazy
    valloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)  # Mini-batche po 32 obrazy
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)  # Mini-batche po 32 obrazy

    # Inicjalizacja sieci, funkcji kosztu oraz optymalizatora z nowym współczynnikiem uczenia
    net = EnhancedCNN(num_classes=10)  # Tworzymy instancję sieci, teraz dla 10 klas
    criterion = nn.CrossEntropyLoss()  # Funkcja kosztu (CrossEntropyLoss dla klasyfikacji wieloklasowej)
    optimizer = optim.Adam(net.parameters(), lr=0.0005)  # Optymalizator Adam z niskim współczynnikiem uczenia

    # Funkcja do trenowania modelu
    def train_model():
        train_losses, val_losses = [], []
        for epoch in range(10):  # Liczba epok
            running_loss = 0.0
            net.train()  # Przełączamy sieć na tryb treningowy
            for inputs, labels in trainloader:
                optimizer.zero_grad()  # Zerowanie gradientów
                outputs = net(inputs)  # Propagacja w przód
                loss = criterion(outputs, labels)  # Obliczanie straty
                loss.backward()  # Propagacja wsteczna
                optimizer.step()  # Aktualizacja wag
                running_loss += loss.item()

            train_losses.append(running_loss / len(trainloader))

            # Ewaluacja na zbiorze walidacyjnym po każdej epoce
            val_loss = 0.0
            net.eval()  # Ustawiamy sieć w tryb ewaluacji (brak propagacji wstecznej)
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

if __name__ == '__main__':
    main()
