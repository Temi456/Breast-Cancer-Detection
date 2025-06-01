# -*- coding: utf-8 -*-
"""

@author: TemiA
"""

#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# -----------------------------
# Custom dataset class with targeted augmentations
# This class loads image-label pairs from a CSV and applies different transforms 
# based on whether the image belongs to the minority class or not.
# -----------------------------
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, base_transform=None, minority_transform=None):
        self.data = pd.read_csv(csv_file)  # Read the CSV file containing image filenames and labels
        self.root_dir = root_dir            # Directory where images are stored
        self.base_transform = base_transform
        self.minority_transform = minority_transform

    def __len__(self):
        return len(self.data)  # Total number of samples

    def __getitem__(self, idx):
        # Build image file path
        img_name = os.path.join(self.root_dir, f"{self.data.iloc[idx, 0]}.pgm")
        
        # Open image in grayscale, then convert to RGB (3 channels) for pretrained model compatibility
        image = Image.open(img_name).convert("L")
        image = image.convert("RGB")
        
        label = int(self.data.iloc[idx, 1])  # Extract label
        
        # Apply stronger augmentation for minority class (label 2), otherwise base augmentation
        if label == 2 and self.minority_transform:
            image = self.minority_transform(image)
        elif self.base_transform:
            image = self.base_transform(image)

        return image, label

# -----------------------------
# Define the augmentations for majority and minority classes
# Majority classes get milder augmentation, minority class gets stronger augmentations
# -----------------------------
base_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                         std=[0.229, 0.224, 0.225])   # ImageNet std
])

minority_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Paths to image data and labels CSV
# -----------------------------
image_dir = "C:\\Users\\tobil\\Downloads\\all-mias\\MIAS DATASET"
label_file = "C:\\Users\\tobil\\Downloads\\Mias spreadsheet.csv"

# Initialize dataset with our transforms
dataset = CustomDataset(label_file, image_dir, base_transform=base_transform, minority_transform=minority_transform)

# -----------------------------
# Split the dataset into train, validation, and test sets (70/10/20 split)
# -----------------------------
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

# -----------------------------
# Create a weighted sampler to balance classes during training
# This oversamples the minority class to prevent bias
# -----------------------------
def get_sampler_weights(dataset_subset):
    # Get all labels in the subset
    labels = [dataset_subset[i][1] for i in range(len(dataset_subset))]
    
    # Count samples per class
    class_sample_counts = np.bincount(labels)
    
    # Prevent division by zero if any class is missing
    class_sample_counts = np.array([count if count > 0 else 1 for count in class_sample_counts])
    
    # Inverse frequency as weights
    weights = 1. / class_sample_counts
    
    # Assign weight to each sample according to its class
    sample_weights = [weights[label] for label in labels]
    return sample_weights

train_weights = get_sampler_weights(train_data)

train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)

batch_size = 32

# Data loaders for training (with sampler), validation and test
train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# -----------------------------
# Load pretrained ResNet18 and customize final layers
# Freeze early layers except last block and classifier to speed training & avoid overfitting
# -----------------------------
model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False  # Freeze all layers initially

for param in model.layer4.parameters():
    param.requires_grad = True   # Unfreeze last residual block

# Replace the classifier head with dropout + linear layer for 3 classes
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_features, 3)
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# -----------------------------
# Set up loss function, optimizer and learning rate scheduler
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# -----------------------------
# Training loop
# Train for specified epochs, compute train and validation losses and accuracies
# Adjust learning rate based on validation loss
# -----------------------------
num_epochs = 10
train_losses, val_losses = [], []
train_acc, val_acc = [], []

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss_train, correct_train, total_train = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        running_loss_train += loss.item()

    train_loss = running_loss_train / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(train_loss)
    train_acc.append(train_accuracy)

    # Validate on validation set
    model.eval()  # Set model to evaluation mode
    running_loss_val, correct_val, total_val = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            running_loss_val += loss.item()

    val_loss = running_loss_val / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    val_losses.append(val_loss)
    val_acc.append(val_accuracy)

    # Step the scheduler based on validation loss
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

# -----------------------------
# Plot training and validation loss curves
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# -----------------------------
# Plot training and validation accuracy curves
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

# -----------------------------
# Final evaluation on the test set
# Compute predictions, confusion matrix and classification report
# -----------------------------
all_labels, all_preds = [], []
model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix on Test Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("Classification Report on Test Set:\n", classification_report(all_labels, all_preds))
