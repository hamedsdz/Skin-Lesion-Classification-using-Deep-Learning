import os
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
from .dataset import ISICDataset, get_transforms

class ISICDataManager:
    def __init__(self, config):
        self.config = config
        self.dataset_root = Path(config['data']['dataset_root'])
        
    def combine_datasets(self):
        """Combine 2017 and 2018 datasets"""
        # 2017 Dataset paths
        train_2017 = {
            'images': self.dataset_root / '2017/ISIC-2017_Training_Data.zip',
            'labels': self.dataset_root / '2017/ISIC-2017_Training_Part3_GroundTruth.csv'
        }
        
        val_2017 = {
            'images': self.dataset_root / '2017/ISIC-2017_Validation_Data.zip',
            'labels': self.dataset_root / '2017/ISIC-2017_Validation_Part3_GroundTruth.csv'
        }
        
        test_2017 = {
            'images': self.dataset_root / '2017/ISIC-2017_Test_v2_Data.zip',
            'labels': self.dataset_root / '2017/ISIC-2017_Test_v2_Part3_GroundTruth.csv'
        }
        
        # 2018 Dataset paths
        train_2018 = {
            'images': self.dataset_root / '2018/ISIC2018_Task3_Training_Input.zip',
            'labels': self.dataset_root / '2018/ISIC2018_Task3_Training_GroundTruth.zip'
        }
        
        val_2018 = {
            'images': self.dataset_root / '2018/ISIC2018_Task3_Validation_Input.zip',
            'labels': self.dataset_root / '2018/ISIC2018_Task3_Validation_GroundTruth.zip'
        }
        
        test_2018 = {
            'images': self.dataset_root / '2018/ISIC2018_Task3_Test_Input.zip',
            'labels': self.dataset_root / '2018/ISIC2018_Task3_Test_GroundTruth.zip'
        }
        
        # Create datasets
        train_transform = get_transforms(self.config, is_train=True)
        val_transform = get_transforms(self.config, is_train=False)
        
        # Training datasets
        train_datasets = []
        if train_2017['images'].exists():
            train_datasets.append(
                ISICDataset(train_2017['images'], train_2017['labels'], 
                           transform=train_transform, year=2017)
            )
        if train_2018['images'].exists():
            train_datasets.append(
                ISICDataset(train_2018['images'], train_2018['labels'], 
                           transform=train_transform, year=2018)
            )
            
        # Validation datasets
        val_datasets = []
        if val_2017['images'].exists():
            val_datasets.append(
                ISICDataset(val_2017['images'], val_2017['labels'], 
                           transform=val_transform, year=2017)
            )
        if val_2018['images'].exists():
            val_datasets.append(
                ISICDataset(val_2018['images'], val_2018['labels'], 
                           transform=val_transform, year=2018)
            )
            
        # Test datasets
        test_datasets = []
        if test_2017['images'].exists():
            test_datasets.append(
                ISICDataset(test_2017['images'], test_2017['labels'], 
                           transform=val_transform, year=2017)
            )
        if test_2018['images'].exists():
            test_datasets.append(
                ISICDataset(test_2018['images'], test_2018['labels'], 
                           transform=val_transform, year=2018)
            )
        
        return train_datasets, val_datasets, test_datasets
