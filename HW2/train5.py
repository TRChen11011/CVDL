import torch
import torch.nn as nn
import torchvision
import os
from torchvision import datasets, transforms
import numpy as np
import cv2
from PIL import Image
import torch.optim as optim
import torchsummary as summary
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomErasing(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if __name__ == '__main__':

    train_dir = "./training_dataset"
    test_dir = "./validation_dataset"
    trainset = datasets.ImageFolder(root=train_dir, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = datasets.ImageFolder(root=test_dir, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    classes = trainset.classes
    print(classes)

    resnet50=torchvision.models.resnet50()
    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    resnet50 = resnet50.to(device)

    print('Resnet50')
    print(num_ftrs)
    criterion = nn.BCELoss()  
    optimizer = optim.Adam(resnet50.parameters(), lr=0.0015) 
    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []

    num_epochs = 30
    best_val_loss = float("inf")


    for epoch in range(num_epochs):
        print(f"Epoch {epoch +1}/{num_epochs}")
        run_loss=0.0
        correct_train=0
        total_train=0
        resnet50.train()

        for i,data in enumerate(trainloader):
            if i%50==49:
                print(i)
            
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.to(device)
                labels = labels.to(device)
                
            outputs = resnet50(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            run_loss += loss.item()   
            if (i+1) %50 == 0:
                print(f"Epoch: {epoch+1}, [{(i+1)*(len(inputs))}/{(len(trainset))}]")
            
            predicted = (outputs > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels.unsqueeze(1).float()).sum().item()
        
        train_accuracy.append(100 * correct_train / total_train)
        train_loss.append(run_loss / len(trainloader))
        
        print(f"Train Loss: {train_loss[-1]:.4f}")
        print(f"Train Accuracy: {train_accuracy[-1]:.2f}%")
        
        resnet50.eval()
        correct_val = 0
        total_val = 0
        run_loss_val = 0.0
        
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                outputs = resnet50(inputs)
                
                loss = criterion(outputs.squeeze(), labels.float())
                run_loss_val += loss.item()
                
                predicted = (outputs > 0.5).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels.unsqueeze(1).float()).sum().item()

        
        val_accuracy.append(100 * correct_val / total_val)
        val_loss.append(run_loss_val / len(testloader))
        
        print(f"Validation Loss: {val_loss[-1]:.4f}")
        print(f"Validation Accuracy: {val_accuracy[-1]:.2f}%")
        

    print('training is done')
    print('wait to save the module')        
    np.savetxt('train_loss.txt', train_loss)
    np.savetxt('val_loss.txt', val_loss)

    np.savetxt('train_accuracy.txt', train_accuracy)
    np.savetxt('val_accuracy.txt', val_accuracy)

    torch.save(resnet50, 'ResNet50_catdog_ereasing_weights.pth')