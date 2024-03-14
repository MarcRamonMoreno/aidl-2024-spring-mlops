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
        # Assuming the dataframe has columns 'file' for image filenames and 'label' for labels
        self.data = self.labels_df['file'].values
        self.labels = self.labels_df['label'].values

        

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        img_name = os.path.join(self.images_path, self.data[idx])
        image = Image.open(img_name)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

            return image, label