# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 15:32:06 2025

@author: szk9
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:02:44 2025

@author: szk9
"""



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import numpy as np
import gaussiand2D_layer_pytorch as gauss
from ptflops import get_model_complexity_info


## Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## The hyperparameters for the model
BATCH = 32
EPOCH = 100
classes = 10
kernel_size = 3

## Loading the data
#data = h5py.File("C:/Combined/Work/Gauss-2D/dataset_h5/data.h5", 'r')
data = h5py.File("C:/Combined/Work/Gauss-2D/dataset_testing/codes/h5/my_usg.h5", 'r')
x = np.asarray(data['dataset_gray']).astype('float32') / 255.0
y = np.asarray(data['dataset_label'])

## The train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

## Convert data to PyTorch tensors
x_train = torch.tensor(x_train).unsqueeze(1).squeeze(-1).to(device)
x_test = torch.tensor(x_test).unsqueeze(1).squeeze(-1).to(device)

y_train = torch.tensor(y_train).long().to(device)
y_test = torch.tensor(y_test).long().to(device)

## Convert y_train to a 1D NumPy array before counting unique elements
unique_train_samples = len(np.unique(y_train.cpu().numpy()))
unique_test_samples = len(np.unique(y_test.cpu().numpy()))

## Data loader
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=BATCH, shuffle=True,  drop_last=True )
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=BATCH,  drop_last=True )


## Defining the PyTorch model
class MyModel(nn.Module):
    def __init__(self, num_classes=classes):
        super(MyModel, self).__init__()

        
        # Block 1
        self.block1 = nn.Sequential(gauss.GaussNetLayer2D(32, (kernel_size, kernel_size)),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Block 4
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

# Gflops calculation
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


## Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)



## Training loop starts here.......................................................................................
train_losses = []
val_losses = []
best_accuracy = 0.0  # Track best accuracy
best_model_path = "C:/Combined/Work/Gauss-2D/saved_finals/best_model.pth"


for epoch in range(EPOCH):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        labels = labels.squeeze()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    
    print(f"Epoch [{epoch+1}/{EPOCH}], Loss: {train_loss:.4f}")

    
## Evaluating the model
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            labels = labels.squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    val_loss /= len(test_loader)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    test_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCH}], Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    
    # **SAVE THE BEST MODEL**
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), best_model_path)
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        print(f"New Best Model Saved! Accuracy: {best_accuracy:.2f}%")

## LOAD THE BEST MODEL AFTER TRAINING
model.load_state_dict(torch.load(best_model_path, weights_only=True))
model.to(device)
model.eval()
print(f"Loaded the Best Model with Accuracy: {best_accuracy:.2f}%")


## Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCH + 1), train_losses, label='Training Loss')
plt.plot(range(1, EPOCH + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.show()

## Confusion matrix visualization
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[f"Class {i}" for i in range(classes)], yticklabels=[f"Class {i}" for i in range(classes)])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

## the and classification report
class_report = classification_report(y_true, y_pred, target_names=[f"Class {i}" for i in range(classes)])
print("Classification Report:")
print(class_report)
