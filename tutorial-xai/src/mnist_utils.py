import json
import pandas as pd
import numpy as np
import zipfile
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy

def unzip_data(zip_path, extract_dir):
    # Create the folder if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)

    # Unzip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

class MNISTData(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).view(-1, 1, 28, 28)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_prep_data(path):
    data = pd.read_csv(path)  # [19999, 785]

    # Data preperation
    X = data.iloc[:, 1:].values.astype(np.float32) / 255.0
    y = data.iloc[:, 0].values.astype(np.int64)

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = MNISTData(X_train, y_train)
    val_dataset = MNISTData(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader

# Implement CNN
class LitMNIST_CNN(pl.LightningModule):
    def __init__(self, lr=0.001):
        super().__init__()
        self.save_hyperparameters()

        # Define model architecture
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

        # Define loss and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

def fit_model(model, train_loader, val_loader):
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_loader)

def load_test_data(path):
    test_data = pd.read_csv(path)  # [9999, 785]

    X_test = test_data.iloc[:, 1:].values.astype("float32") / 255.0
    y_test = test_data.iloc[:, 0].values.astype("int64")

    test_dataset = MNISTData(X_test, y_test)
    return test_dataset

def plot_gradients(input, vanilla_gradient, grad_x_input):
    plt.figure(figsize=(12, 4))  # wider to fit 3 images

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(input.cpu().detach().numpy().squeeze(), cmap='gray')
    plt.title(f"Original image")
    plt.axis('off')

    # Gradient importance
    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(vanilla_gradient), cmap='hot')
    plt.title("Vanilla Gradients")
    plt.axis('off')

    #  Gradient × Input importance
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(grad_x_input), cmap='hot')
    plt.title("Gradient × Input")
    plt.axis('off')

    plt.show()