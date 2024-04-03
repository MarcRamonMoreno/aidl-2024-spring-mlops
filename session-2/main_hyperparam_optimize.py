import ray
import torch
from ray import tune
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.dataset import random_split
from dataset import MyDataset
from model import MyModel
from utils import accuracy, save_model



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        print(f"Output shape: {outputs.shape}, Target shape: {labels.shape}")
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_correct += accuracy(outputs, labels)
    
    return total_loss / len(data_loader)




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
    
    return total_loss / len(data_loader)


def train_model(config):
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    my_model = MyModel().to(device)
    optimizer = optim.Adam(my_model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["epochs"]):
        train_loss = train_single_epoch(my_model, train_loader, optimizer, criterion, device)
        val_loss = eval_single_epoch(my_model, val_loader, criterion, device)
        tune.report(loss=val_loss, accuracy=0)  # Example metric, adjust as necessar)


def test_model(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    test_loss = eval_single_epoch(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss}")



if __name__ == "__main__":

    # Example transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Corrected normalization
    ])


    my_dataset = MyDataset(images_path='/home/mramon/Escritorio/AI_Deep_Learning_UPC/MLOPs/aidl-2024-spring-mlops/session-2/data/data', 
                           labels_path='/home/mramon/Escritorio/AI_Deep_Learning_UPC/MLOPs/aidl-2024-spring-mlops/session-2/chinese_mnist.csv', 
                           transform=transform)
    # Assuming my_dataset is already defined
    dataset_size = len(my_dataset)
    train_size = int(dataset_size * 0.7)
    val_size = int(dataset_size * 0.15)
    test_size = dataset_size - (train_size + val_size)
    train_dataset, val_dataset, test_dataset = random_split(my_dataset, [train_size, val_size, test_size])
    ray.init(configure_logging=False, num_gpus=1, local=True)
    analysis = tune.run(
        train_model,
        resources_per_trial={"cpu": 1, "gpu": 1},
        metric="val_loss",
        mode="min",
        num_samples=5,
        config={
            "hyperparam_1": tune.uniform(1, 10),  # Assuming this is used somewhere in your model or training process
            "hyperparam_2": tune.grid_search(["relu", "tanh"]),  # Assuming this influences the activation function in your model
            "batch_size": tune.choice([16, 32, 64, 128]),  # Adding batch_size to the config
            "lr": tune.loguniform(1e-4, 1e-1),  # Example of how to add learning rate if it's not already there
            "epochs": 10  # Assuming a fixed number of epochs for simplicity; could also be made variable
        }
    )

    print("Best hyperparameters found were: ", analysis.best_config)
    # Assuming we're directly using the best model for simplicity,
    # though typically you'd reload or reconstruct it with the best hyperparameters
    # and possibly train it on the combined train+val set before testing.
    best_trial = analysis.best_trial
    best_model = MyModel(activation=best_trial.config["activation"]).to(device)
    # Assuming save_model function saves and loads model correctly.
    save_model(best_model, f"{best_trial.logdir}/best_model.pth")
    best_model.load_state_dict(torch.load(f"{best_trial.logdir}/best_model.pth"))

    # You'd typically reconstruct the model and optimizer here with the best hyperparameters
    # and possibly retrain on the full training set (train + val) before testing.
    test_model(best_model, test_dataset)