import os
import yaml
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import sys

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.data_manager import ISICDataManager
from src.models.model import create_model
from src.utils.metrics_manager import MetricsManager
from src.utils.logger import setup_logging

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Setup logging and metrics
        self.setup_logging()
        self.metrics_manager = MetricsManager(config)
        
        # Setup data
        self.setup_data()
        
        # Create model
        self.model = create_model(config).to(self.device)
        
        # Setup loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=config['training']['scheduler']['patience'],
            factor=config['training']['scheduler']['factor']
        )
        
        # Initialize best metrics for model checkpointing
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def setup_logging(self):
        """Setup logging and tensorboard"""
        self.log_dir = Path(self.config['logging']['log_dir'])
        self.save_dir = Path(self.config['logging']['save_dir'])
        
        self.log_dir.mkdir(exist_ok=True)
        self.save_dir.mkdir(exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        setup_logging(self.log_dir / 'training.log')
        self.logger = logging.getLogger(__name__)
        
    def setup_data(self):
        """Setup datasets and dataloaders"""
        data_manager = ISICDataManager(self.config)
        train_datasets, val_datasets, test_datasets = data_manager.combine_datasets()
        
        # Combine datasets
        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = ConcatDataset(val_datasets)
        self.test_dataset = ConcatDataset(test_datasets)
        
        print(f"Number of training samples: {len(self.train_dataset)}")
        print(f"Number of validation samples: {len(self.val_dataset)}")
        print(f"Number of test samples: {len(self.test_dataset)}")
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers']
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers']
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers']
        )
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Store predictions and labels for metrics
            pred = output.argmax(dim=1).cpu().numpy()
            target = target.argmax(dim=1).cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(target)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate and store metrics
        metrics = self.metrics_manager.update_metrics(
            np.array(all_labels),
            np.array(all_preds),
            epoch,
            'train'
        )
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    def validate(self, epoch, phase='val'):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        loader = self.val_loader if phase == 'val' else self.test_loader
        
        with torch.no_grad():
            for data, target in tqdm(loader, desc=phase.capitalize()):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                # Store predictions and labels for metrics
                pred = output.argmax(dim=1).cpu().numpy()
                target = target.argmax(dim=1).cpu().numpy()
                all_preds.extend(pred)
                all_labels.extend(target)
        
        # Calculate and store metrics
        metrics = self.metrics_manager.update_metrics(
            np.array(all_labels),
            np.array(all_preds),
            epoch,
            phase
        )
        metrics['loss'] = total_loss / len(loader)
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save latest checkpoint
        checkpoint_path = self.save_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Training on device: {self.device}")
        
        for epoch in range(self.config['training']['epochs']):
            # Training phase
            train_metrics = self.train_epoch(epoch)
            self.logger.info(f"Epoch {epoch} training metrics: {train_metrics}")
            
            # Log training metrics
            for name, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'train/{name}', value, epoch)
            
            # Validation phase
            val_metrics = self.validate(epoch, 'val')
            self.logger.info(f"Epoch {epoch} validation metrics: {val_metrics}")
            
            # Log validation metrics
            for name, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'val/{name}', value, epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config['training']['early_stopping']['patience']:
                self.logger.info("Early stopping triggered")
                break
        
        # Final evaluation on test set
        test_metrics = self.validate(epoch, 'test')
        self.logger.info(f"Final test metrics: {test_metrics}")
        
        # Plot and export results
        self.metrics_manager.plot_training_curves()
        self.metrics_manager.plot_confusion_matrix(
            np.array(all_labels),
            np.array(all_preds),
            self.config['classes']
        )
        self.metrics_manager.export_results()
        
        self.logger.info("Training completed!")
        self.writer.close()

def main():
    # Load configuration
    with open('src/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
