import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images_path = images_path
        self.labels_df = pd.read_csv(labels_path)
        # If no transform is specified, default to converting images to tensors
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        suite_id, sample_id, code, value, character = self.labels_df.loc[idx]
        img_name = os.path.join(self.images_path, f"input_{suite_id}_{sample_id}_{code}.jpg")

        if not os.path.exists(img_name):
            # Handle missing files appropriately here
            print(f"Warning: File {img_name} not found.")
            return None  # Adjust based on your error handling preferences

        image = Image.open(img_name)

        # Apply the transformation pipeline, including converting to tensor
        image = self.transform(image)

        return image, code - 1
