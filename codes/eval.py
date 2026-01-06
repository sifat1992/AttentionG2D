# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 10:53:09 2025

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
from dataset import H5AugmentedDataset  
import argparse

classes=10

# ----------------------------------------------------------
# Arguments
# ----------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (gauss, gauss_se, custom, mobilenet, resnet50, dense121, alex, vgg16, eca)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="C:/Combined/Work/Gauss_no_data_leakage/path/gauss_best.pth")
    parser.add_argument("--batch", type=int, default=32)
    return parser.parse_args()


# ----------------------------------------------------------
#Evaluation function
# ----------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)

            preds.extend(predicted.cpu().numpy())
            trues.extend(labels.cpu().numpy())

    # Metrics
    acc = accuracy_score(trues, preds) * 100
    prec = precision_score(trues, preds, average="weighted", zero_division=0)
    rec = recall_score(trues, preds, average="weighted", zero_division=0)
    f1 = f1_score(trues, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(trues, preds)

    return acc, prec, rec, f1, cm


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n Evaluating model: {args.model} on {device}\n")

    # ------------------------------------------------------
    # Loading Radiologist-2 dataset
    # ------------------------------------------------------
    data = h5py.File("C:/Combined/Work/Gauss_no_data_leakage/h5/my_test.h5", "r") #========

    x_test = np.asarray(data["dataset_gray"]).squeeze()
    y_test = np.asarray(data["dataset_label"])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])


    test_dataset = H5AugmentedDataset(x_test, y_test, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    
    
    
    
        # -----------------------------
    # DEBUG PRINTS
    # -----------------------------
    # print(">>> TEST FILE: C:/Combined/Work/Gauss_no_data_leakage/h5/radio_test.h5")
    # print(">>> TEST SHAPE:", x_test.shape)
    # print(">>> LABEL DISTRIBUTION:", np.unique(y_test, return_counts=True))
    # # -----------------------------

    # -----------------------------

    # ------------------------------------------------------
    # Loading model
    # ------------------------------------------------------
    model = get_model(args.model, num_classes=classes).to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    # ------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------
    acc, prec, rec, f1, cm = evaluate(model, test_loader, device)

    print("\n================ Evaluation Results ================")
    print(f"Accuracy:  {acc:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("====================================================\n")


if __name__ == "__main__":
    main()
