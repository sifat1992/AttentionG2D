# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 11:05:06 2025

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
from sklearn.preprocessing import label_binarize
import warnings
import os
from sklearn.exceptions import UndefinedMetricWarning
import gaussiand2D_layer_pytorch as gauss
train_on_gpu = torch.cuda.is_available()
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)



## Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## The hyperparameters for the model

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
classes = 10
LR=0.001
LR_THRESHOLD = 1e-5 
weight_decay=0.001
kernel_size = 3

#=================================================================

## Loading the data
# data = h5py.File("C:/Combined/Work/Gauss-2D/dataset_testing/codes/h5/my_usg.h5", 'r')
# x = np.asarray(data['dataset_gray'])
# y = np.asarray(data['dataset_label'])

## Loading the data
data = h5py.File("C:/Combined/Work/Gauss-2D/dataset_testing/codes/h5/my_usg.h5", 'r')
x = np.asarray(data['dataset_gray']).astype('float32') / 255.0
y = np.asarray(data['dataset_label'])

x_tensor = torch.tensor(x[:, :, :, 0], dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.long).squeeze()


# x_tensor = torch.tensor(x[:, :, :, 0], dtype=torch.float32).unsqueeze(1)
# y_tensor = torch.tensor(y, dtype=torch.long).squeeze()

train_dataset = TensorDataset(x_tensor, y_tensor)
indices_data = list(range(len(train_dataset))) 

## Creating the model architecture<
class MyModel(nn.Module):
    def __init__(self, num_classes=classes):
        super(MyModel, self).__init__()

     

        self.block1 = nn.Sequential(gauss.GaussNetLayer2D(32, (kernel_size, kernel_size)),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Block 1
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Block 2
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Block 3
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 1 * 1, 256)  
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        #x = self.gauss_layer(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    


## Instantiate the model
model = MyModel().to(device)
print(model)

#=============================================cross validation================================================================
num_data = len(train_dataset)
indices_data = list(range(num_data))
ss = ShuffleSplit(n_splits=5, test_size=0.20, random_state=42)
# for train_index, test_index in ss.split(indices_data):
#   print("\n%s %s" % (len(train_index), len(test_index)))
#   print("Training: %s \nTesting: %s" % (train_index, test_index))


## Training loop starts here.................................................................................
fold_accuracies = []
for fold, (train_index, test_index) in enumerate(ss.split(indices_data), start=1):
    print("\n" + "="*60)
    print(f"🚀 Fold {fold} / 5")
    print("="*60)

    # --- Data split for this fold ---
    train_sampler = SubsetRandomSampler(train_index)
    test_sampler = SubsetRandomSampler(test_index)

    train_loader = DataLoader(train_dataset, batch_size=BATCH, sampler=train_sampler,  drop_last=True)
    test_loader = DataLoader(train_dataset, batch_size=BATCH,  drop_last=True, sampler=test_sampler)

    # --- Model, loss, optimizer, scheduler ---
    model = MyModel(num_classes=classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
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