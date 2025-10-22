# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 10:00:57 2025

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
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
from ptflops import get_model_complexity_info
import torch.nn as nn
from torchsummary import summary
import torchvision.models as models
import torch.nn as nn


## Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


## The hyperparameters for the model
BATCH = 32
EPOCH = 80
classes = 10
kernel_size = 3


class MobileNetV1(nn.Module):
    def __init__(self, ch_in, n_classes):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            #conv_dw(256, 512, 2),
            #conv_dw(512, 512, 1),
            #conv_dw(512, 512, 1),
            #conv_dw(512, 512, 1),
            #conv_dw(512, 512, 1),
            #conv_dw(512, 512, 1),
            #conv_dw(512, 1024, 2),
            #conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

if __name__=='__main__':
    # model check
    model = MobileNetV1(ch_in=1, n_classes=classes).to(device)
    #summary(model, input_size=(3, 224, 224), device='cpu')

# \Gflops calculation
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



## Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


## Loading the data
data = h5py.File("C:/Combined/Work/Gauss-2D/dataset_testing/codes/h5/my_usg.h5", 'r')
x = np.asarray(data['dataset_gray']).astype('float32') / 255.0
y = np.asarray(data['dataset_label'])

## The train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x = np.asarray(data['dataset_gray']).astype('float32') / 255.0
y = np.asarray(data['dataset_label'])

## The train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Convert data to PyTorch tensors
x_train = torch.tensor(x_train).unsqueeze(1).squeeze(-1).to(device)  # Adding channel dimension and removing last dimension
x_test = torch.tensor(x_test).unsqueeze(1).squeeze(-1).to(device)

y_train = torch.tensor(y_train).long().to(device)

y_test = torch.tensor(y_test).long().to(device)

# Data loader
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), drop_last=True, batch_size=BATCH, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), drop_last=False, batch_size=BATCH)

## Function to count the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

## Display model parameters
num_params = count_parameters(model)
print(f"Total number of parameters: {num_params}")


## Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

## Training loop starts here.......................................................................................
train_losses = []
val_losses = []
best_accuracy = 0.0  # Track best accuracy
best_model_path = "C:/Combined/Work/Gauss-2D/ieee_mod/best_mods/mobilenet1.pth"


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
    y_probs = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            labels = labels.squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            #_, predicted = torch.max(outputs, 1)
            #total += labels.size(0)
            probs = torch.softmax(outputs, dim=1)  # use sigmoid if binary
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())  

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
model.load_state_dict(torch.load(best_model_path,weights_only=True))
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

# Plot
num_classes = classes
y_true_bin = label_binarize(y_true, classes=list(range(classes)))
y_probs = np.array(y_probs)  # shape should be [num_samples, num_classes]

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], lw=2,
              label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
