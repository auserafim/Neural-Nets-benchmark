import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Transformações
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_emnist = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Configurações
dataset_name = 'emnist'  # 'cifar10', 'cifar100' ou 'emnist'
model_name = 'resnet50'  # 'resnet50' ou 'efficientnet_b0'
num_epochs = 10
batch_size = 64
learning_rate = 1e-4

# Dataset
if dataset_name == 'cifar10':
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    num_classes = 10
elif dataset_name == 'cifar100':
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    num_classes = 100
elif dataset_name == 'emnist':
    train_dataset = datasets.EMNIST(root='./data', split="mnist", train=True, download=True, transform=transform_emnist)
    test_dataset = datasets.EMNIST(root='./data', split="mnist", train=False, download=True, transform=transform_emnist)
    num_classes = 10
else:
    raise ValueError("Escolha inválida para dataset_name.")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Modelo
if model_name == 'resnet50':
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
elif model_name == 'efficientnet_b0':
    model = models.efficientnet_b0(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
else:
    raise ValueError("Escolha inválida para model_name.")

# Treino
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

print(f"Modelo {model_name} carregado com {num_classes} classes e usando {device}.")

loss_per_epoch = []

# Início do tempo
start_time = time.time()

# Loop de treinamento
s = 0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    loss_per_epoch.append(epoch_loss)
    print(f"Época {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")
    s += epoch_loss

# Fim do tempo
end_time = time.time()
training_time = end_time - start_time

# Plot da loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs+1), loss_per_epoch, marker='o')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.title(f'Training Loss - {model_name.upper()} - {dataset_name.upper()}')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'loss_{model_name}_{dataset_name}.png')
#plt.show()
s = s / num_epochs
print("loss eh:", s)
# Avaliação
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Métricas
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

# Exibe no terminal
print(f"Acurácia: {accuracy*100:.2f}%")
print(f"Precisão: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1-score: {f1*100:.2f}%")
print(f"Tempo de treinamento: {training_time:.2f} segundos")

# Salvar modelo
model_path = f'{model_name}_{dataset_name}.pth'
torch.save(model.state_dict(), model_path)
print(f"Modelo salvo como {model_path}")

