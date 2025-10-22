# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 13:58:53 2025

@author: szk9
"""

import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.model_selection import ShuffleSplit
from torch.utils.data.sampler import SubsetRandomSampler

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import h5py
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import torch.nn.functional as F

train_on_gpu = torch.cuda.is_available()

#============reprodtivility===================
seed = 4051
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#===========================================================


#===================hyperparameters=============================
BATCH = 32
EPOCH = 60
classes = 4
LR=1e-5
LR_THRESHOLD = 1e-5 
weight_decay=0.005
#=================================================================

## Load dataset
data = h5py.File("C:/Combined/Work/Gauss-2D/dataset_testing/codes/h5/data_ct.h5", 'r')
x = np.asarray(data['dataset_gray'])
y = np.asarray(data['dataset_label'])

x_tensor = torch.tensor(x[:, :, :, 0], dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.long).squeeze()

train_dataset = TensorDataset(x_tensor, y_tensor)
indices_data = list(range(len(train_dataset))) 

# The bottlneck layers
class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BottleneckLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], 1)  ## Concatenate the input and the output
    
## Transition layers are used to connect dense blocks to reducing the number of feature maps and downsampling the spatial dimensions of the feature maps.
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out

## The denseblocks are the building blocks of DenseNet architectures
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(n_layers):
            layers.append(BottleneckLayer(in_channels + i * growth_rate, growth_rate))
        self.denseblock = nn.Sequential(*layers)

    def forward(self, x):
        return self.denseblock(x)

## Assembling the densenet model
class DenseNet(nn.Module):
    def __init__(self, block_config=(6, 12, 24, 16), growth_rate=32, num_classes=classes):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        self.conv1 = nn.Conv2d(1, 2 * growth_rate, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * growth_rate)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        num_channels = 2 * growth_rate
        self.dense_layers = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_channels, growth_rate, num_layers)
            self.dense_layers.append(block)
            num_channels += num_layers * growth_rate

            if i != len(block_config) - 1:
                transition = TransitionLayer(num_channels, num_channels // 2)
                self.dense_layers.append(transition)
                num_channels = num_channels // 2

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        for layer in self.dense_layers:
            x = layer(x)

        x = F.relu(self.bn2(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

## Example of how to create a DenseNet model (among DenseNet-121, DenseNet-169, DenseNet-201, and DenseNet-264, we will use densenet121)
def densenet121(num_classes=classes):
    return DenseNet(block_config=(6, 12, 24, 16), growth_rate=32, num_classes=classes)

## Instantiate model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenseNet(block_config=(6,12,24,16), num_classes=classes).to(device)

## Check for GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")



# ## Function to count the number of parameters in the model
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ## Display model parameters
# num_params = count_parameters(model)
# print(f"Total number of parameters: {num_params}")
# print (model)

#=============================================cross validation================================================================
num_data = len(train_dataset)
indices_data = list(range(num_data))
ss = ShuffleSplit(n_splits=5, test_size=0.20, random_state=42)
# for train_index, test_index in ss.split(indices_data):
#   print("\n%s %s" % (len(train_index), len(test_index)))
#   print("Training: %s \nTesting: %s" % (train_index, test_index))

## Training loop starts here.......................................................................................................
fold_accuracies = []

for fold, (train_index, test_index) in enumerate(ss.split(indices_data), start=1):
    print("\n" + "="*60)
    print(f"🚀 Fold {fold} / 5")
    print("="*60)

    # --- Data split for this fold ---
    train_sampler = SubsetRandomSampler(train_index)
    test_sampler = SubsetRandomSampler(test_index)

    train_loader = DataLoader(train_dataset, batch_size=BATCH, sampler=train_sampler)
    test_loader = DataLoader(train_dataset, batch_size=BATCH, sampler=test_sampler)

    # --- Model, loss, optimizer, scheduler ---
    model = DenseNet(block_config=(6, 12, 24, 16), growth_rate=32, num_classes=classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )

    # ===================================================
    #  Training
    # ===================================================
    for epoch in range(EPOCH):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # ---- Validation per epoch ----
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(test_loader)
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCH}] | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.2f}%")

        scheduler.step(avg_val_loss)
        if optimizer.param_groups[0]["lr"] < LR_THRESHOLD:
            print(f"Learning rate dropped below {LR_THRESHOLD}. Early stopping.")
            break

    # ===================================================
    #  Fold Evaluation
    # ===================================================
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    acc = np.mean(np.array(y_pred) == np.array(y_true)) * 100
    fold_accuracies.append(acc)

    print(f"\n✅ Fold {fold} Accuracy: {acc:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

# ===================================================
# 🏁 Summary after all folds
# ===================================================
avg_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)

print("\n" + "="*60)
print("🎯 FINAL CROSS-VALIDATION RESULTS")
print("="*60)
for i, acc in enumerate(fold_accuracies, start=1):
    print(f"Fold {i}: {acc:.2f}%")
print(f"\nAverage Accuracy: {avg_acc:.2f}%")
print(f"Standard Deviation: {std_acc:.2f}%")