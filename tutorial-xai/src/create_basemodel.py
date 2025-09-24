"""
This script creates our BASEMODEL!
Therefore it uses a pre-trained ResNet-18 model and fine-tunes it for 50 epochs on Cifar-10 dataset.

Leading to a accuracy on test dataset of 95.85%

Input: Pre-Trained ResNet-18
Output: BASEMODEL
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import submitit

from utils_summerschool import *


if __name__ == "__main__":
    # -----------------------------
    # Load CIFAR-10 
    # -----------------------------
    transform_train = transforms.Compose([
        transforms.Resize(224),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                            std=[0.229, 0.224, 0.225])
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                            shuffle=False, num_workers=2)

    # -----------------------------
    # Load Pretrained ResNet-18
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # CIFAR-10 has 10 classes
    model = model.to(device)

    # -----------------------------
    # Loss, Optimizer and Scheduler
    # -----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # -----------------------------
    # Training-Loop
    # -----------------------------
    for epoch in range(50): 
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")
        
        scheduler.step()

    print("Training done.")

    # -----------------------------
    # Evaluation 
    # -----------------------------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test set: {100 * correct / total:.2f}%")

    torch.save(model, "models/Basemodel_Summerschool.pth")


