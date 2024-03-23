import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import MyDataset  # Adjust import path as necessary
from model import MyModel
from utils import accuracy, save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_single_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_correct += accuracy(outputs, labels)
    
    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_correct / len(data_loader.dataset)
    print(f"Train Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}")

def eval_single_epoch(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            total_correct += accuracy(outputs, labels)
    
    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_correct / len(data_loader.dataset)
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}")


def train_model(config, dataset_paths):
    # Load dataset
    full_dataset = MyDataset(dataset_paths['images'], dataset_paths['labels'])
    
    # Split dataset into train, validation, and test
    train_size = int(0.7 * len(full_dataset))  # 70% of dataset for training
    val_size = int(0.15 * len(full_dataset))  # 15% of dataset for validation
    test_size = len(full_dataset) - (train_size + val_size)  # Remaining 15% for testing
    
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model = MyModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        train_single_epoch(model, train_loader, optimizer, criterion, device)
        eval_single_epoch(model, val_loader, criterion, device)  # Evaluate on validation set
    
    # Optionally evaluate on test set after training and validation are complete
    # eval_single_epoch(model, test_loader, criterion, device)



# Assuming train_single_epoch and eval_single_epoch are defined similarly to before, but with appropriate logging/messages for clarity.

if __name__ == "__main__":
    config = {
        "epochs": 10,
        "batch_size": 64,
        "learning_rate": 0.001,
        # Include other configurations as necessary
    }
    dataset_paths = {
        "images": "/home/marc/Escritorio/UPC_AI_Deep_Learning/MLOps/aidl-2024-spring-mlops/session-2/data/data",  # Update this path
        "labels": "/home/marc/Escritorio/UPC_AI_Deep_Learning/MLOps/aidl-2024-spring-mlops/session-2/chinese_mnist.csv",  # Update this path
    }
    train_model(config, dataset_paths)
                            