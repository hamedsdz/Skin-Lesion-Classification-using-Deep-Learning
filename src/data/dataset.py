import os
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ISICDataset(Dataset):
    """ISIC Skin Lesion Dataset"""
    
    def __init__(self, zip_path, labels_path, transform=None, year=2018):
        """
        Args:
            zip_path (str): Path to the zip file containing images
            labels_path (str): Path to the labels file (zip for 2018, csv for 2017)
            transform (callable, optional): Optional transform to be applied on a sample
            year (int): Dataset year (2017 or 2018)
        """
        self.zip_path = Path(zip_path)
        self.labels_path = Path(labels_path)
        self.transform = transform
        self.year = year
        
        # Load and process labels
        self.labels_df = self._load_labels()
        self.image_ids = list(self.labels_df.index)
        
        # Verify zip file exists
        if not self.zip_path.exists():
            raise FileNotFoundError(f"Images zip file not found: {self.zip_path}")
        
        print(f"Loaded {len(self.image_ids)} samples from {year} dataset")
        
    def _load_labels(self):
        """Load labels based on dataset year"""
        if self.year == 2018:
            with zipfile.ZipFile(self.labels_path) as z:
                # Find the CSV file in the zip
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                if not csv_files:
                    raise ValueError("No CSV file found in labels zip")
                
                # Read the first CSV file found
                with z.open(csv_files[0]) as f:
                    df = pd.read_csv(f, index_col=0)
        else:  # 2017
            df = pd.read_csv(self.labels_path)
            # Convert image_id to index
            df.set_index('image_id', inplace=True)
            
            # For 2017, we have binary labels for melanoma and seborrheic_keratosis
            # Convert to one-hot format with 7 classes (matching 2018 format)
            labels = np.zeros((len(df), 7))
            
            # Set melanoma (class 0)
            labels[df['melanoma'] == 1.0, 0] = 1
            
            # Set seborrheic keratosis (class 2)
            labels[df['seborrheic_keratosis'] == 1.0, 2] = 1
            
            # Set nevus (class 1) for cases that are neither melanoma nor seborrheic keratosis
            nevus_mask = (df['melanoma'] == 0.0) & (df['seborrheic_keratosis'] == 0.0)
            labels[nevus_mask, 1] = 1
            
            # Create new dataframe with one-hot encoded labels
            df = pd.DataFrame(
                labels,
                index=df.index,
                columns=['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC']
            )
        
        return df
    
    def _load_image(self, image_id):
        """Load image from zip file"""
        with zipfile.ZipFile(self.zip_path) as z:
            # Handle different naming conventions between 2017 and 2018
            if self.year == 2018:
                image_pattern = image_id
            else:
                image_pattern = image_id  # Already in ISIC_XXXXXXX format
            
            # Find the image file in the zip
            image_files = [f for f in z.namelist() if image_pattern in f]
            if not image_files:
                raise ValueError(f"Image {image_id} not found in zip")
            
            # Read the image file
            with z.open(image_files[0]) as f:
                img = Image.open(f).convert('RGB')
        return img
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        image = self._load_image(image_id)
        image = np.array(image)
        
        # Get labels
        labels = self.labels_df.loc[image_id].values.astype(np.float32)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, labels

def get_transforms(config, is_train=True):
    """Get transforms for training or validation"""
    if is_train:
        return A.Compose([
            A.Resize(height=config['data']['image_size'][0],
                    width=config['data']['image_size'][1]),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
                A.ColorJitter(p=1)
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(p=1),
                A.GaussianBlur(p=1),
                A.ISONoise(p=1)
            ], p=0.2),
            A.Normalize(
                mean=config['augmentation']['train']['normalize']['mean'],
                std=config['augmentation']['train']['normalize']['std']
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=config['data']['image_size'][0],
                    width=config['data']['image_size'][1]),
            A.Normalize(
                mean=config['augmentation']['val']['normalize']['mean'],
                std=config['augmentation']['val']['normalize']['std']
            ),
            ToTensorV2(),
        ])
