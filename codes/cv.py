# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 14:42:59 2025

@author: szk9
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from dataset import H5AugmentedDataset
from architectures import get_model
import torchvision.transforms as transforms
import argparse
import warnings

warnings.filterwarnings("ignore")

classes=4

# =============================================
# CLI arguments
# =============================================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--save", type=str,
                        default="C:/Combined/Work/Gauss_no_data_leakage/cv_checkpoints")
    return parser.parse_args()


# =============================================
# Per-model LR + WD from my train.py, keeping the same structure and all.
# =============================================
MODEL_LR = {
    "gauss": 0.001, "gauss_se": 0.001, "custom": 8e-5, "mobilenet": 0.001,
    "resnet50": 1e-4, "dense121": 1e-5, "alex": 1e-5, "vgg16": 0.001, "eca": 0.001
}

WEIGHT_DECAY = {
    "gauss": 0.001, "gauss_se": 0.001, "custom": 0.002, "mobilenet": 0.001,
    "resnet50": 1e-4, "dense121": 0.005, "alex": 0.005, "vgg16": 1e-4, "eca": 0.001
}


# =============================================
# Train one epoch
# =============================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        _, preds = torch.max(out, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return loss_sum / len(loader), 100 * correct / total


# =============================================
# MAIN CV
# =============================================
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n🔥 Running {args.folds}-Fold Cross Validation on {args.model}\n")

    # Load HDF5 dataset
    data = h5py.File("C:/Combined/Work/Gauss_no_data_leakage/h5/ct_train.h5", "r")
    x = np.asarray(data["dataset_gray"]).squeeze()
    y = np.asarray(data["dataset_label"])

    dataset = H5AugmentedDataset(x, y, transforms.ToTensor())
    indices = np.arange(len(dataset))
    ss = ShuffleSplit(n_splits=args.folds, test_size=0.2, random_state=42)

    # Storage for fold metrics
    fold_acc = []
    fold_pre = []
    fold_rec = []
    fold_f1  = []

    # =========================================
    # FOLDS
    # =========================================
    for fold, (train_idx, val_idx) in enumerate(ss.split(indices), start=1):

        print(f"\n====================== Fold {fold} ======================")

        train_loader = DataLoader(dataset, batch_size=args.batch,
                                  sampler=SubsetRandomSampler(train_idx),
                                  drop_last=True)

        val_loader = DataLoader(dataset, batch_size=args.batch,
                                sampler=SubsetRandomSampler(val_idx),
                                drop_last=False)

        # Model
        model = get_model(args.model, num_classes=classes).to(device)
        criterion = nn.CrossEntropyLoss()

        lr = MODEL_LR.get(args.model, 1e-3)
        wd = WEIGHT_DECAY.get(args.model, 0)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # Train
        for ep in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader,
                                                    criterion, optimizer, device)
            print(f"Epoch {ep+1}/{args.epochs}  Train Acc: {train_acc:.2f}%")

        # =====================================
        # Validation for current fold
        # =====================================
        y_true, y_pred = [], []

        model.eval()
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                out = model(x_b)
                _, preds = torch.max(out, 1)

                y_true.extend(y_b.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred, average="macro")
        rec = recall_score(y_true, y_pred, average="macro")
        f1  = f1_score(y_true, y_pred, average="macro")

        fold_acc.append(acc)
        fold_pre.append(pre)
        fold_rec.append(rec)
        fold_f1.append(f1)

        print(f"\nFold {fold} Results:")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {pre:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")

        # Save model
        torch.save(model.state_dict(),
                   f"{args.save}/{args.model}_fold{fold}.pth")

    # ================================================
    # PRINT ALL FOLDS TOGETHER + AVERAGES
    # ================================================
    print("\n====================== 5-FOLD RESULTS ======================\n")
    
    for i in range(args.folds):
        print(f"Fold {i+1}:")
        print(f"  Accuracy : {fold_acc[i]:.4f}")
        print(f"  Precision: {fold_pre[i]:.4f}")
        print(f"  Recall   : {fold_rec[i]:.4f}")
        print(f"  F1 Score : {fold_f1[i]:.4f}\n")
    
    print("------------------------ AVERAGES --------------------------\n")
    
    print(f"Mean Accuracy : {np.mean(fold_acc):.4f}   Std: {np.std(fold_acc):.4f}")
    print(f"Mean Precision: {np.mean(fold_pre):.4f}   Std: {np.std(fold_pre):.4f}")
    print(f"Mean Recall   : {np.mean(fold_rec):.4f}   Std: {np.std(fold_rec):.4f}")
    print(f"Mean F1 Score : {np.mean(fold_f1):.4f}    Std: {np.std(fold_f1):.4f}")
    
    print("\n============================================================")


if __name__ == "__main__":
    main()
