import torch
from torch.utils.data import DataLoader

from model import MyModel
from utils import binary_accuracy_with_logits
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(model, train_loader, optimizer, loss_function):
    model.train()
    accs, losses = [], []
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        y = y.unsqueeze(-1).float()  # Adjust label shape and ensure it's float

        optimizer.zero_grad()

        outputs = model(x)

        loss = loss_function(outputs, y)
        loss.backward()
        optimizer.step()

        acc = binary_accuracy_with_logits(y, outputs)
        losses.append(loss.item())
        accs.append(acc)

    return np.mean(losses), np.mean(accs)


def eval_single_epoch(model, test_loader, loss_function):
    model.eval()
    accs, losses = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device).float().unsqueeze(-1)  # Ensure matching shape for y
            outputs = model(x)
            loss = loss_function(outputs, y)
            # Assuming you have a binary_accuracy function suitable for your adjusted model output
            acc = binary_accuracy_with_logits(y, outputs)  # Ensure this function matches your model's output
            losses.append(loss.item())
            accs.append(acc)
    return np.mean(losses), np.mean(accs)


def train_model(config):

    # Data transformations for training and testing datasets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    train_dataset = ImageFolder(root='dataset/cars_vs_flowers/training_set', transform=data_transforms['train'])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    test_dataset = ImageFolder(root='dataset/cars_vs_flowers/test_set', transform=data_transforms['test'])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    my_model = MyModel().to(device)
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(my_model.parameters(), lr=config["lr"])
    for epoch in range(config["epochs"]):
        loss, acc = train_single_epoch(my_model, train_loader, optimizer, loss_function)
        print(f"Train Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
        loss, acc = eval_single_epoch(my_model, test_loader, loss_function)
        print(f"Eval Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")
    
    return my_model


if __name__ == "__main__":

    config = {
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 5,
    }
    my_model = train_model(config)

    
