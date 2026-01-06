# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 14:14:07 2022

@author: sb3682
"""

import numpy as np
import os
import cv2
import h5py

main_folder_path = 'C:/Combined/Work/Gauss_no_data_leakage/radiologist_1/test'

label = []
data = []
i = 0
for class_folder in os.listdir(main_folder_path):
    class_folder_path = os.path.join(main_folder_path, class_folder)
    
    if os.path.isdir(class_folder_path):
        print(f"Current folder: {class_folder}")
        for img_name in os.listdir(class_folder_path):
            img_path = os.path.join(class_folder_path, img_name)
            imgray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Fix for reading the image
            
            if imgray is not None:  # Ensure the image was read successfully
                data_gray = cv2.resize(imgray, (256, 256))
                label.append(int(i))  # Append the label, no assignment needed
                data.append(data_gray)  # Append the resized grayscale image
            else:
                print(f"Error reading image: {img_name}")
    
    i += 1  # Increment the label index for the next class

# Convert data and label to numpy arrays
data = np.expand_dims(np.asarray(data),axis=-1)
label = np.expand_dims(np.asarray(label),axis=-1)

hf = h5py.File('C:/Combined/Work/Gauss_no_data_leakage/h5/radio_test.h5', 'w')
hf.create_dataset('dataset_gray', data=data)
hf.create_dataset('dataset_label', data=label)
hf.close()