import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.nns.Net import CustomDataset, SimpleCNN, validate_model
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split
import copy

num_model = 7
batch_size = 8
num_epochs = 1000

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
dataset = CustomDataset(data_dir='../stitches', transform=data_transform)

total_size = len(dataset)
# Počet obrázků pro trénování a validaci
train_size = int(0.8 * total_size)
valid_size = total_size - train_size
# Rozdělení datasetu
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))


# Vytvoření DataLoaderů
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=16, shuffle=False)

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
best_loss = 100000000
best_acc = 0
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
    val_loss, val_accuracy = validate_model(model, criterion, valid_loader)
    if val_loss < best_loss:
        best_loss_model = copy.deepcopy(model)
        best_loss_epoch = epoch+1
        best_loss = val_loss
    if val_accuracy > best_acc:
        best_acc_model = copy.deepcopy(model)
        best_acc_epoch = epoch+1
        best_acc = val_accuracy

    if epoch % 10 == 9:  # Tisk průběžné ztráty každých 10 epoch
        print('[%d] train loss: %.3f' %
              (epoch + 1, running_loss / 10))
        running_loss = 0.0
        print(f'[{epoch+1}] Validation Loss:    \t{val_loss:.4f}')
        print(f'[{epoch+1}] Validation Accuracy:\t{val_accuracy:.2f}%')
        print("")
    losses.append(val_loss)
    accuracys.append(val_accuracy)


torch.save(best_acc_model.state_dict(), 'models//best_acc_2model'+str(num_model)+'_'+str(best_acc_epoch)+'.pth')
torch.save(best_loss_model.state_dict(), 'models//best_loss_2model'+str(num_model)+'_' + str(best_loss_epoch) + '.pth')

torch.save(model.state_dict(), 'models//final_model_'+str(num_model)+'_' + str(best_loss_epoch) + '.pth')


losses_csv = 'train_data//losses2_'+str(num_model)+'.csv'
accuracys_csv = 'train_data//accuracys2_'+str(num_model)+'.csv'


losses_csv_arr = np.asarray(losses)
accuracys_csv_arr = np.asarray(accuracys)

np.savetxt(losses_csv, losses_csv_arr, delimiter=",")
np.savetxt(accuracys_csv, accuracys_csv_arr, delimiter=",")

# # Uložení seznamu do CSV souboru
# with open(losses_csv, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(losses)

# with open(accuracys_csv, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(accuracys)

print('Finished Training')




