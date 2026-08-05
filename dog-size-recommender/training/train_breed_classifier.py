"""
training/train_breed_classifier.py
------------------------------------
Trains the ResNet18 breed classifier (Golden Retriever vs Labrador Retriever)
and saves it to models/breed_classifier.pth, matching the format
src/predict_breed.py expects.

Folder layout expected:
    data/breed_images/
        train/
            golden_retriever/*.jpg
            labrador_retriever/*.jpg
        val/
            golden_retriever/*.jpg
            labrador_retriever/*.jpg

Run from the project root:
    python -m training.train_breed_classifier
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model(num_classes=2, unfreeze_last_block=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    if unfreeze_last_block:
        for param in model.layer4.parameters():
            param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def train(data_dir="data/breed_images", epochs=10, batch_size=32, lr=1e-3, save_path="models/breed_classifier.pth"):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print(f"ERROR: expected {train_dir} and {val_dir} to exist with class subfolders.")
        print("See this file's docstring for the expected layout.")
        sys.exit(1)

    train_ds = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_transforms)

    print("Detected classes (check this matches src/predict_breed.py CLASS_NAMES order):", train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(train_ds.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_ds)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total else 0.0

        print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} - val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({"model_state": model.state_dict(), "classes": train_ds.classes}, save_path)
            print(f"  -> saved new best model (val_acc={val_acc:.4f}) to {save_path}")

    print(f"Training done. Best val_acc={best_val_acc:.4f}. Saved to {save_path}")


if __name__ == "__main__":
    train()
