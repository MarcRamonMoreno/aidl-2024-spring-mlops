import os

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image



class MyDataset(Dataset):

    def __init__(self, images_path, labels_path, transform=None):
        self.images_path = images_path
        self.transform = transform
        # Load labels
        self.labels_df = pd.read_csv(labels_path)


        

    def __len__(self):
        return len(self.labels_df)


    def __getitem__(self, idx):
        suite_id, sample_id, code, value, character = self.labels_df.loc[idx, :]
        img_name = os.path.join(self.images_path, f"input_{suite_id}_{sample_id}_{code}.jpg")
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

            return image, code-1