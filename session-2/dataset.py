import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images_path = images_path
        self.labels_df = pd.read_csv(labels_path)
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Example values for grayscale
        ])
        
        # Filter out rows where the image file does not exist
        self.labels_df = self.labels_df[self.labels_df.apply(lambda row: os.path.exists(
            os.path.join(images_path, f"input_{row['suite_id']}_{row['sample_id']}_{row['code']}.jpg")), axis=1)]

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_name = os.path.join(self.images_path, f"input_{row['suite_id']}_{row['sample_id']}_{row['code']}.jpg")
        
        image = Image.open(img_name)
        image = self.transform(image)
        
        # Assuming 'code' is the label and subtracting 1 to make labels zero-indexed
        return image, row['code'] - 1
