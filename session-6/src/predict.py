from typing import List

import torch

from model import RegressionModel


@torch.no_grad()
def predict(input_features: List[float]):
    # Load the checkpoint from the correct path
    checkpoint_path = '/checkpoints/checkpoint.pt'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Instantiate the model and load the state dict
    input_size = checkpoint['input_size']
    hidden_size = checkpoint['hidden_size']
    output_size = 1  # Assuming output size is 1 for regression
    model = RegressionModel(input_size, hidden_size, output_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    
    # Convert input features to a tensor of the correct shape
    x = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)
    
    # Retrieve normalization parameters
    x_mean = checkpoint['x_mean']
    x_std = checkpoint['x_std']
    y_mean = checkpoint['y_mean']
    y_std = checkpoint['y_std']
    
    # Normalize the input features using the training parameters
    x = (x - x_mean) / x_std
    
    # Get the output of the model
    output = model(x)
    
    # Revert the target normalization that was done during training
    output = output * y_std + y_mean
    
    # Print the result, adjusting if you scaled your target differently during training
    print(f"The predicted price is: ${output.item():.2f}")