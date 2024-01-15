import sys
import os
import numpy as np
import PIL
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

if __name__ == '__main__':

    batch = 64
    num_epochs = 35

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=False)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch, shuffle=False, num_workers=8)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = torchvision.models.vgg19_bn(num_classes=10)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0015)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        count=0
        for inputs, labels in train_loader:
            count = count+1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            if count%100==99:
                print(f'[{epoch + 1}, {count + 1:5d}] loss: {running_loss / 100:.3f}')
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(test_loader)
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs}, '
            f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
            f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    torch.save(model, 'my_trained_model_vgg19_Q4.pth')
    torch.save(model.state_dict(), 'my_model_weights_vgg19_Q4.pth')
    print('Finished Training')

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.savefig('vgg19_bn_training_results.png')
    plt.show()