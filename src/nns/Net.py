import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = os.listdir(data_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_name).convert('L')  # Převod na černobílý obrázek
        # label = int(self.image_files[idx][-5])  # Extrahování labelu ze jména souboru
        label = int(self.image_files[idx].split('_')[0])  # Extrahování labelu ze jména souboru

        # one_hot_label = torch.zeros(2)
        # one_hot_label[label] = 1
        if self.transform:
            image = self.transform(image)
        return image,label
    def getTransform(self):
        return self.transform


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 10 * 5, 128)  # Přizpůsobeno velikosti výstupu z konvolučních vrstev
        self.fc2 = nn.Linear(128, 2)  # Dvě třídy: steh, nebo ne steh

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32 * 10 * 5)  # Přizpůsobeno velikosti výstupu z konvolučních vrstev
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)  # Pokus
        return x

def validate_model(model, criterion, valid_loader):
    model.eval()  # Přepnutí modelu do evaluačního režimu
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # Deaktivace výpočtu gradientů
        for images, labels in valid_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(valid_loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy

def load_model_SimpleCNN(path_model, n=1):
    model_weights_file = path_model

    model_weights = torch.load(model_weights_file)
    if n == 1:
        model = SimpleCNN()
    elif n == 4:
        model = SimpleCNN_4()
    else:
        print("unidentified n = ", n)
    model.load_state_dict(model_weights)

    return model

def predict(model, img_stitchable, numm=0):

    if numm==0:
        transform = transforms.Compose([
            transforms.Resize((40, 20)),  # Změna velikosti na 40x20
            transforms.ToTensor()  # Převod na PyTorch tensor
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((40, 20)),  # Změna velikosti na 40x20
            transforms.ToTensor(),  # Převod na PyTorch tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalizace
        ])

    img = Image.fromarray(img_stitchable)
    model.eval()
    # Příprava vstupního obrázku k predikci
    # image_path = 'image.jpg'
    # image = Image.open(image_path).convert('L')
    img = transform(img).unsqueeze(0)  # Přidání dimenze batche

    with torch.no_grad():
        output = model(img)
    # output = output.numpy()[0]
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    # print(output)
    # print(probabilities)
    # print("")
    # return int(output[0]>-output[1])
    return predicted_class



import torch
import torch.nn as nn
import torch.optim as optim

# Definice jednoduché CNN
class SimpleCNN_2(nn.Module):
    def __init__(self):
        super(SimpleCNN_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16 * 9 * 4, 1)  # 40x20 -> 20x10 -> 10x5

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 9 * 4)
        x = torch.sigmoid(self.fc1(x))
        return x

# Načtení dat a trénink

class SimpleCNN_3(nn.Module):
    def __init__(self):
        super(SimpleCNN_3, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 10 * 5, 128)  # Přizpůsobeno velikosti výstupu z konvolučních vrstev
        self.fc2 = nn.Linear(128, 2)  # Dvě třídy: steh, nebo ne steh

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32 * 10 * 5)  # Přizpůsobeno velikosti výstupu z konvolučních vrstev
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)  # Pokus
        return x

class SimpleCNN_4(nn.Module):
    def __init__(self):
        super(SimpleCNN_4, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 10 * 5, 64)
        self.fc2 = nn.Linear(64, 1)  # Jeden neuron na výstupu pro binární klasifikaci

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 16 * 10 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid na výstupu pro binární klasifikaci
        return x


def predict_single_image(skimage_image, model):
    # Transformace pro testovací obrázek
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Převod ze skimage na PIL Image
        transforms.Resize((40, 20)),  # Změna velikosti na 40x20
        transforms.ToTensor(),  # Převod na PyTorch tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalizace
    ])

    # Aplikace transformací na obrázek
    pil_image = transform(skimage_image)
    image = pil_image.unsqueeze(0)

    # print(image)

    # Nastavení modelu do režimu vyhodnocování
    model.eval()

    # S predikcí neprovádíme výpočet gradientů
    with torch.no_grad():
        # Provádíme predikci
        output = model(image)

    # Získání indexu třídy s nejvyšší pravděpodobností
    predicted_class = torch.sigmoid(output).round().int().item()
    # print(output)
    # print("Predikce:", predicted_class)
    # print("")
    # Pokud máme víc než jednu třídu, můžeme použít:
    # predicted_class = torch.argmax(output).item()

    return predicted_class
