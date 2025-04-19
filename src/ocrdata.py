'''
author: Rupesh Garsondiya
github: @Rupeshgarsondiya
organization: L.J University
'''


import os
import torch
import pandas as pd
import pytorch_lightning as pl

from PIL import Image
from typing import Any
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# Define dataset class
class OCRDataset(Dataset):
    def __init__(self, csv_file_path, img_dir, transform):
        self.data = pd.read_csv(csv_file_path)  # Load CSV file
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        
        Args:
        - (None)

        Returns:
        - (int): Number of sam  labels = [self.label_map[item[1]] for item in batcples in the dataset.
        '''
        return len(self.data)

    def __getitem__(self, idx) -> tuple:
        '''
        Returns a sample from the dataset.

        Args:
        - idx (int): Index of the sample.

        Return:
        - (tuple): Sample from the dataset.
        '''

        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])  # get image path
        label = self.data.iloc[idx, 1]  # get text label
        image = Image.open(img_path).convert("RGB")  # open image

        if self.transform:
            # Define image transformations
            transform_obj = transforms.Compose([
                transforms.Resize((384, 384)),  # Resize to match model input size
                transforms.ToTensor(),  # Convert to tensor
                transforms.Normalize((0.5,), (0.5,))  # normalize
                ])
            
            image = transform_obj(image)

        return image, label  # return image tensor and label


class OCRDataModule(pl.LightningDataModule):
    def __init__(self, csv_file_path, image_dir, transform, batch_size=32) -> None:
        super().__init__()
        self.csv_file = csv_file_path
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.transform = transform
        self.label_map = None  # Store label encoding

    def setup(self, stage=None) -> None:
        '''
        Setup the dataset.

        Args:
        - stage (str): Stage of the training process. Can be 'fit', 'validate','test', or 'predict'.
        
        Returns:
        - None
        '''
        self.train_dataset = OCRDataset(self.csv_file, self.image_dir, self.transform)

        # Step 1: Compute label encoding ONCE
        all_labels = [item[1] for item in self.train_dataset]
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(all_labels)

        # Step 2: Create a mapping {label_text: label_number}
        self.label_map = {label: idx for label, idx in zip(all_labels, encoded_labels)}

    def collate_fn(self, batch):
        '''
        Custom collate function to handle text labels.

        Args:
        - batch (list): List of tuples containing image tensors and labels.

        Returns:
        - tuple: Tuple containing a list of image tensors and a list of encoded labels.
        '''
        # Extract images (tensors) and stack them
        pixel_values = torch.stack([item[0] for item in batch])  

        # Convert string labels to encoded numbers
        labels = [self.label_map[item[1]] for item in batch]  
        labels = torch.tensor(labels, dtype=torch.long).unsqueeze(1)  # Ensure shape is (batch_size, seq_len)

        return pixel_values, labels

    def train_dataloader(self) -> DataLoader:
        '''
        Returns the training dataloader.

        Args:
        - None

        Returns:
        - DataLoader: Training dataloader.
        '''
        print("Training data size:", len(self.train_dataset))
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)