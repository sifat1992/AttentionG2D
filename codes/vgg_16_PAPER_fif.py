# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 12:48:55 2025

@author: szk9
"""

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
from ptflops import get_model_complexity_info

## Hyperparameters
BATCH = 32
EPOCH = 60
classes = 4

## Load dataset
data = h5py.File("C:/Combined/Work/Gauss-2D/dataset_testing/codes/h5/data_ct.h5", 'r')
x = np.asarray(data['dataset_gray'])
y = np.asarray(data['dataset_label'])

## The train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

## Convert data to torch tensors
x_train = torch.tensor(x_train[:, :, :, 0], dtype=torch.float32).unsqueeze(1)  # Remove last dimension and add channel dimension for grayscale
x_test = torch.tensor(x_test[:, :, :, 0], dtype=torch.float32).unsqueeze(1)  # Remove last dimension and add channel dimension for grayscale

y_train = torch.tensor(y_train, dtype=torch.long).squeeze()
y_test = torch.tensor(y_test, dtype=torch.long).squeeze()

## Create DataLoader for PyTorch
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)

## Define the VGG-like network architecture
class CustomVGG(nn.Module):
    def __init__(self, num_classes=classes):
        super(CustomVGG, self).__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

       
        # Adjust the following line accordingly:
        self.fc1 = nn.Linear(128 * 10 * 10, 512)
        
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

## Instantiate model, define loss and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomVGG(classes).to(device)

#Gflops calculation
macs, params = get_model_complexity_info(
    model, (1, 256, 256),
    as_strings=False,  
    print_per_layer_stat=False,
    verbose=False
)

gflops = (2 * macs) / 1e9  
print(f"GFLOPs: {gflops:.2f}")
print(f"MACs: {macs:,}")
print(f"Parameters: {params:,}")


## Function to count the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

## Display model parameters
num_params = count_parameters(model)
print(f"Total number of parameters: {num_params}")
print (model)

## Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

## Training loop starts here.................................................................................
for epoch in range(EPOCH):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        ## Zero the parameter gradients
        optimizer.zero_grad()
        
        ## Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        ## Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  ## Gradient clipping
        optimizer.step()
        
        running_loss += loss.item()

    ## Compute average training loss
    #avg_loss = running_loss / len(train_loader)
    train_loss = running_loss / len(train_loader)

    ## Step the scheduler with validation loss
    #scheduler.step(train_loss)

    print(f"Epoch [{epoch+1}/{EPOCH}] | Train Loss: {train_loss:.4f}")

    ## Evaluate model on test set at the end of each epoch
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    avg_val_loss = val_loss / len(test_loader)
    val_acc = 100 * correct / total
    
    # Step scheduler with validation loss
    scheduler.step(avg_val_loss)
    print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    
print(f"Epoch [{epoch+1}/{EPOCH}] | Train Loss: {train_loss:.4f} | "
      f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
## Evaluate model performance with metrics
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[f"Class {i}" for i in range(classes)], yticklabels=[f"Class {i}" for i in range(classes)])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

## classification report
class_report = classification_report(y_true, y_pred, target_names=[f"Class {i}" for i in range(classes)])
print("Classification Report:")
print(class_report)