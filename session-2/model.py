import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Assuming the images are grayscale, but adjust in_channels if they're not
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # You'll need to adjust the size here based on the size of your images
        # For example, if your images are 64x64, after 3 pools they are 8x8
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)  # Adjust the output features according to the dataset

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        # Flatten the output for the linear layer
        x = x.view(-1, 128 * 8 * 8)  # Adjust this size accordingly
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
