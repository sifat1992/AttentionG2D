# -*- coding: utf-8 -*-
"""
Created on Tue May 27 16:31:58 2025

@author: szk9
"""



import os
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
from torch.nn import init
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


## Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#SE==============================================================================================================

class SEAttention(nn.Module):
    def __init__(self, channel,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

## The hyperparameters for the model
BATCH = 32
EPOCH = 100
classes = 10
kernel_size = 3

## Loading the data
data = h5py.File("C:/Combined/Work/Gauss-2D/dataset_testing/codes/h5/my_usg.h5", 'r')
x = np.asarray(data['dataset_gray']).astype('float32') / 255.0
y = np.asarray(data['dataset_label'])

## The train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



# Define the Dataset class
class H5AugmentedDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx].squeeze()  # [H, W]
        image = (image * 255).astype('uint8')  # Convert to uint8
        image = Image.fromarray(image)         # Convert to PIL image

        if self.transform:
            image = self.transform(image)

        #label = int(self.y[idx])
        label = self.y[idx].item()  # already an int
        return image, label



train_transform = transforms.Compose([
    #transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # must come last!
])

val_transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = H5AugmentedDataset(x_train, y_train, transform=train_transform)
test_dataset = H5AugmentedDataset(x_test, y_test, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32,drop_last=True, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, drop_last=True,shuffle=False)


## Convert y_train to a 1D NumPy array before counting unique elements
unique_train_samples = len(np.unique(y_train))
unique_test_samples = len(np.unique(y_test))

## Defining the PyTorch model
class MyModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        
        #SEblock=======================================================================================================
        self.se1 = SEAttention(64)
        self.se2 = SEAttention(64)
        self.se3 = SEAttention(128)
        self.se4 = SEAttention(128)

        
        #Block 1
        self.block1 = nn.Sequential( gauss.GaussNetLayer2D(64, (kernel_size, kernel_size)),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 1 * 1, 256)  
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        #x = self.gauss_layer(x)
        x = self.block1(x)
        x = self.se1(x)
        x = self.block2(x)
        x = self.se2(x)
        x = self.block3(x)
        x = self.se3(x)
        x = self.block4(x)
        x = self.se4(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    


## Instantiate the model
model = MyModel().to(device)
print(model)

# \Gflops calculation
macs, params = get_model_complexity_info(
    model, (1, 256, 256),
    as_strings=False,   # gives raw numbers
    print_per_layer_stat=False,
    verbose=False
)

gflops = (2 * macs) / 1e9  # FLOPs ≈ 2×MACs
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
best_accuracy = 0.0 
best_model_path = "C:/Combined/Work/Gauss-2D/dataset_testing/codes/best_model_se_64_att.pth"


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
        
# saving vector in csv files 
np.savetxt("C:/Combined/Work/Gauss-2D/se64_train_loss_vector.csv", np.array(train_losses), delimiter=",")
np.savetxt("C:/Combined/Work/Gauss-2D/se64_val_loss_vector.csv", np.array(val_losses), delimiter=",")
    
               
        
## LOAD THE BEST MODEL AFTER TRAINING
model.load_state_dict(torch.load(best_model_path,weights_only=True))
model.to(device)
model.eval()
print(f"Loaded the Best Model with Accuracy: {best_accuracy:.2f}%")


# plt.figure(figsize=(8, 6))  
plt.plot(range(1, EPOCH + 1), train_losses, label='Training Loss', linewidth=2.5)
plt.plot(range(1, EPOCH + 1), val_losses, label='Validation Loss', linewidth=2.5)

plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training and Validation Loss over Epochs', fontsize=16)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)

# Save as PDF (vector format - perfect for LaTeX)
plt.tight_layout()
plt.savefig("training_validation_loss.pdf", format='pdf', dpi=300)
plt.show()



## Confusion matrix visualization
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[f"Class {i}" for i in range(classes)], yticklabels=[f"Class {i}" for i in range(classes)])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()



# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Save confusion matrix to CSV
np.savetxt("fif_plots/confusion_matrix_s.csv", conf_matrix, fmt='%d', delimiter=',')

# Plot it in style
plt.figure(figsize=(12, 10))
sns.set(font_scale=1.4)

ax = sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='YlGnBu',
    cbar=True,
    xticklabels=[f"Class {i}" for i in range(classes)],
    yticklabels=[f"Class {i}" for i in range(classes)],
    annot_kws={"size": 12, "weight": "bold"}
)

plt.xlabel("Predicted Labels", fontsize=16, weight='bold')
plt.ylabel("True Labels", fontsize=16, weight='bold')
plt.title("Confusion Matrix", fontsize=18, weight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# # Ensure directory exists
os.makedirs("fif_plots", exist_ok=True)

#Save as high-res PDF
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/confusion_matrix.pdf", format='pdf', dpi=300)
plt.show()

## the and classification report
class_report = classification_report(y_true, y_pred, target_names=[f"Class {i}" for i in range(classes)])
print("Classification Report:")
print(class_report)

# Roc Auc
num_classes = 10
y_true_bin = label_binarize(y_true, classes=list(range(10)))
y_probs = np.array(y_probs) 

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

plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Multiclass ROC Curve', fontsize=16, weight='bold')
plt.grid(True)


plt.legend(
    loc='center left', 
    bbox_to_anchor=(1.02, 0.5), 
    borderaxespad=0,
    fontsize=12
)

plt.tight_layout(rect=[0, 0, 0.85, 1]) 

# Save to folder
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/roc_curve.pdf", format='pdf', dpi=300, bbox_inches='tight')
plt.show()
