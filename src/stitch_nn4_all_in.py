import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.nns.Net import CustomDataset, SimpleCNN
import torch.optim as optim
import torch.nn as nn

num_model = 7
batch_size = 16
num_epochs = 800

# Definice transformací
data_transform = transforms.Compose([
    transforms.Resize((40, 20)),  # Změna velikosti na 40x20
    transforms.RandomVerticalFlip(),  # Náhodné převrácení svisle
    transforms.RandomHorizontalFlip(),  # Náhodné převrácení vodorovně
    transforms.RandomRotation(180),  # Náhodné otočení o 180 stupňů
    transforms.ToTensor(),  # Převod na PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalizace
])

# Vytvoření instance datasetu
train_dataset = CustomDataset(data_dir='../stitches', transform=data_transform)

# Vytvoření DataLoaderů
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
# Kontrola funkčnosti DataLoaderu
for images, labels in train_loader:
    print(images.shape, labels)
    break
print("")
model = SimpleCNN()

# Nastavení kritéria (loss function) a optimalizátoru
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("cuda available: ", torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

losses = []
accuracys = []
model.to(device)
# Trénovací smyčka
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)


        # print("inputs: ",inputs.shape)
        # print("outputs: ", outputs.shape)
        # print("labels: ", labels.shape)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if epoch % 10 == 9:  # Tisk průběžné ztráty každých 10 epoch
        print('[%d] train loss: %.3f' %
              (epoch + 1, running_loss / 10))



torch.save(model.state_dict(), 'models//model_all_in_'+str(800)+'.pth')


losses_csv_arr = np.asarray(losses)
accuracys_csv_arr = np.asarray(accuracys)

# # Uložení seznamu do CSV souboru
# with open(losses_csv, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(losses)

# with open(accuracys_csv, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(accuracys)

print('Finished Training')




