import os
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
import gc
import logging
from datetime import datetime
import json
from torch.utils.tensorboard import SummaryWriter

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.model import create_model
from src.data.dataset import ISICDataset, get_transforms
from src.utils.metrics_manager import MetricsManager
from src.evaluate.test import evaluate

class TrainingManager:
    def __init__(self, config):
        self.config = config
        self.setup_directories()
        self.setup_logging()
        self.setup_device()
        self.setup_data()
        self.setup_model()
        self.metrics_manager = MetricsManager(config)
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = ['logs', 'models', 'results', 'models/checkpoints']
        for dir_name in dirs:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            
    def setup_logging(self):
        """Setup logging configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/training_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter(f"logs/tensorboard_{timestamp}")
        
    def setup_device(self):
        """Setup computing device"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        if self.device.type == 'cpu':
            # Set PyTorch to run in deterministic mode for CPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    def setup_data(self):
        """Setup datasets and dataloaders"""
        transforms_train = get_transforms(self.config, is_train=True)
        transforms_val = get_transforms(self.config, is_train=False)
        
        # Load training datasets
        train_datasets = []
        for year in [2017, 2018]:
            try:
                dataset = ISICDataset(
                    self.config['data'][f'train_{year}']['images'],
                    self.config['data'][f'train_{year}']['labels'],
                    transform=transforms_train,
                    year=year
                )
                train_datasets.append(dataset)
                self.logger.info(f"Loaded {year} training dataset: {len(dataset)} samples")
            except Exception as e:
                self.logger.warning(f"Could not load {year} training dataset: {e}")
        
        # Load validation datasets
        val_datasets = []
        for year in [2017, 2018]:
            try:
                dataset = ISICDataset(
                    self.config['data'][f'val_{year}']['images'],
                    self.config['data'][f'val_{year}']['labels'],
                    transform=transforms_val,
                    year=year
                )
                val_datasets.append(dataset)
                self.logger.info(f"Loaded {year} validation dataset: {len(dataset)} samples")
            except Exception as e:
                self.logger.warning(f"Could not load {year} validation dataset: {e}")
        
        # Create dataloaders
        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset(train_datasets),
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data']['pin_memory']
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset(val_datasets),
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data']['pin_memory']
        )
        
    def setup_model(self):
        """Setup model, criterion, optimizer and scheduler"""
        self.model = create_model(self.config).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config['training']['scheduler']['patience'],
            factor=self.config['training']['scheduler']['factor']
        )
        
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, 'models/checkpoints/latest.pth')
        
        # Save periodic checkpoint
        if epoch % self.config['checkpoint']['save_freq'] == 0:
            torch.save(checkpoint, f'models/checkpoints/epoch_{epoch}.pth')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, 'models/best_model.pth')
            
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch']
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        predictions = []
        targets = []
        
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch}')):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            predictions.extend(output.argmax(dim=1).cpu().numpy())
            targets.extend(target.argmax(dim=1).cpu().numpy())
            
            # Free up memory
            del data, target, output, loss
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
        metrics = self.metrics_manager.update_metrics(
            np.array(targets),
            np.array(predictions),
            epoch,
            'train'
        )
        metrics['loss'] = running_loss / len(self.train_loader)
        
        return metrics
        
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                
                predictions.extend(output.argmax(dim=1).cpu().numpy())
                targets.extend(target.argmax(dim=1).cpu().numpy())
                
                # Free up memory
                del data, target, output, loss
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
        metrics = self.metrics_manager.update_metrics(
            np.array(targets),
            np.array(predictions),
            epoch,
            'val'
        )
        metrics['loss'] = running_loss / len(self.val_loader)
        
        return metrics
        
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        start_epoch = 0
        
        # Resume from checkpoint if specified
        if self.config['checkpoint']['resume']:
            start_epoch = self.load_checkpoint(self.config['checkpoint']['path'])
            
        for epoch in range(start_epoch, self.config['training']['epochs']):
            # Training phase
            train_metrics = self.train_epoch(epoch)
            self.logger.info(f"Epoch {epoch} - Train: {train_metrics}")
            
            # Validation phase
            val_metrics = self.validate(epoch)
            self.logger.info(f"Epoch {epoch} - Validation: {val_metrics}")
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Save checkpoint
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1
                
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if patience_counter >= self.config['training']['early_stopping']['patience']:
                self.logger.info("Early stopping triggered")
                break
                
            # Clear memory
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
        self.logger.info("Training completed!")
        self.writer.close()
        
        # Run final evaluation
        self.logger.info("Running final evaluation...")
        test_results = evaluate(self.config, 'models/best_model.pth')
        self.logger.info(f"Test results: {test_results}")

if __name__ == '__main__':
    # Load configuration
    with open('src/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Start training
    trainer = TrainingManager(config)
    trainer.train()
