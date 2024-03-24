import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Configuration for Convolutional layers remains the same
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Assuming images are 64x64, and after 3 pooling layers they become 8x8
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=15)  # Adjusted for 15 classes

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        print(f"After layer1: {x.shape}")
        x = self.pool(self.relu(self.conv2(x)))
        print(f"After layer2: {x.shape}")
        x = self.pool(self.relu(self.conv3(x)))
        print(f"After layer3: {x.shape}")
        x = x.view(-1, 128 * 8 * 8)  # Flatten the output for the linear layer
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
