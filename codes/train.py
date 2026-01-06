# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 09:23:55 2025

@author: szk9
"""

from architectures import get_model
import torch
import torch.nn as nn
import torch.optim as optim
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
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
from ptflops import get_model_complexity_info
from dataset import H5AugmentedDataset  
import argparse

##==================The hyperparameters for the model==================
classes = 10
#=========================================


# ------------------------------------------------------
# Arguments
# ------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (gauss, gauss_se,, custom, mobilenet, resnet50, dense121, alex, vgg16, eca)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=-1)
    parser.add_argument("--save", type=str, default="C:/Combined/Work/Gauss_no_data_leakage/path")
    return parser.parse_args()

# ------------------------------------------------------
# Training loop
# ------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total * 100
    avg_loss = total_loss / len(loader)
    return avg_loss, acc

# ------------------------------------------------------
# Main training script
# ------------------------------------------------------
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n Training {args.model} on {device}...\n")

    # --------------------------------------------------
    
    MODEL_LR = {
        "gauss": 0.001,
        "gauss_se": 0.001,
        "custom": 8e-5,
        "mobilenet": 0.001,
        "resnet50": 1e-4,
        "dense121": 1e-5,
        "alex": 1e-5,
        "vgg16": 0.001,
        "eca": 0.001
    }
    
    weight_decay = {
        "gauss": 0.001,
        "gauss_se": 0.001,
        "custom": 2e-3,
        "mobilenet": 0.001,
        "resnet50": 0.0001,
        "dense121": 0.005,
        "alex": 0.005,
        "vgg16":0.0001,
        "eca": 0.001
    }
    
    # SCHEDULERS = {
    # "gauss": "plateau",
    # "gauss_se": "plateau",
    # "custom": "steplr",
    # "mobilenet": "plateau",
    # "resnet50": "steplr",
    # "dense121": "cosine",
    # "vgg16": "plateau",
    # "alex": "steplr",
    # }
    
    SCHEDULER_CONFIG = {
    "gauss":       {"factor": 0.5, "patience": 5},
    "gauss_se":    {"factor": 0.5, "patience": 5},
    "custom":      {"factor": 0.1, "patience": 8},
    "mobilenet":   {"factor": 0.5, "patience": 5},
    "resnet50":    {"factor": 0.1, "patience": 3},
    "dense121":    {"factor": 0.1, "patience": 5},
    "alex":        {"factor": 0.1, "patience": 3},
    "vgg16":       {"factor": 0.1, "patience": 3},
    "eca":         {"factor": 0.5, "patience": 5}
    }




    # --------------------------------------------------
    # A. Load HDF5 dataset
    # --------------------------------------------------
    data = h5py.File("C:/Combined/Work/Gauss_no_data_leakage/h5/my_train.h5", "r")
    
    x = np.asarray(data["dataset_gray"])      
    y = np.asarray(data["dataset_label"])
    
    
    x = x.squeeze()        
    
    # --------------------------------------------------
    # Train–test split
    # --------------------------------------------------
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # --------------------------------------------------
    # Transforms
    # --------------------------------------------------
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # --------------------------------------------------
    # Dataset + DataLoader
    # --------------------------------------------------
    train_dataset = H5AugmentedDataset(x_train, y_train, transform=train_transform)
    val_dataset = H5AugmentedDataset(x_val, y_val, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    



    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = get_model(args.model, num_classes=classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Determine LR
    if args.lr < 0:
        lr = MODEL_LR.get(args.model, 1e-3)
    else:
        lr = args.lr
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay.get(args.model, 0)
    )
    
    # building scheduler
    cfg = SCHEDULER_CONFIG.get(args.model, {"factor": 0.5, "patience": 5})

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg["factor"],
        patience=cfg["patience"],
        verbose=True
    )
    
    
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


    # ## Function to count the number of parameters in the model
    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)


    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    best_acc = 0

    for epoch in range(args.epochs):

        # ---- Training for 1 epoch ----
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
    
        # ---- LR scheduler step ----
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_loss)  
            else:
                scheduler.step()
    
        print(f"Epoch {epoch+1}/{args.epochs} — "
              f"Loss: {train_loss:.4f} — Acc: {train_acc:.2f}%")
    
        # ---- Save best checkpoint ----
        if train_acc > best_acc:
            best_acc = train_acc
            save_path = f"{args.save}/{args.model}_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"✔ Saved best checkpoint: {save_path}")
    


if __name__ == "__main__":
    main()


