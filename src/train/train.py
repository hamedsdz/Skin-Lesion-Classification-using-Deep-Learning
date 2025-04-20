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

from src.data.dataset import ISICDataset, get_transforms
from src.models.model import create_model
from src.utils.metrics_manager import MetricsManager
from src.utils.logger import setup_logging

def create_datasets(config, transforms_train, transforms_val):
    """Create datasets for both 2017 and 2018 data"""
    datasets = {
        'train': [],
        'val': [],
        'test': []
    }
    
    # Get dataset root
    dataset_root = Path(config['data']['dataset_root'])
    
    # 2017 Dataset
    try:
        train_2017 = ISICDataset(
            dataset_root / config['data']['train_2017']['images'],
            dataset_root / config['data']['train_2017']['labels'],
            transform=transforms_train,
            year=2017
        )
        datasets['train'].append(train_2017)
        print(f"Loaded 2017 training dataset: {len(train_2017)} samples")
        
        val_2017 = ISICDataset(
            dataset_root / config['data']['val_2017']['images'],
            dataset_root / config['data']['val_2017']['labels'],
            transform=transforms_val,
            year=2017
        )
        datasets['val'].append(val_2017)
        print(f"Loaded 2017 validation dataset: {len(val_2017)} samples")
        
        test_2017 = ISICDataset(
            dataset_root / config['data']['test_2017']['images'],
            dataset_root / config['data']['test_2017']['labels'],
            transform=transforms_val,
            year=2017
        )
        datasets['test'].append(test_2017)
        print(f"Loaded 2017 test dataset: {len(test_2017)} samples")
    except Exception as e:
        print(f"Warning: Could not load 2017 dataset: {e}")
        print(f"Attempted paths:")
        print(f"Train: {dataset_root / config['data']['train_2017']['images']}")
        print(f"Labels: {dataset_root / config['data']['train_2017']['labels']}")
    
    # 2018 Dataset
    # try:
    #     train_2018 = ISICDataset(
    #         dataset_root / config['data']['train_2018']['images'],
    #         dataset_root / config['data']['train_2018']['labels'],
    #         transform=transforms_train,
    #         year=2018
    #     )
    #     datasets['train'].append(train_2018)
    #     print(f"Loaded 2018 training dataset: {len(train_2018)} samples")
        
    #     val_2018 = ISICDataset(
    #         dataset_root / config['data']['val_2018']['images'],
    #         dataset_root / config['data']['val_2018']['labels'],
    #         transform=transforms_val,
    #         year=2018
    #     )
    #     datasets['val'].append(val_2018)
    #     print(f"Loaded 2018 validation dataset: {len(val_2018)} samples")
        
    #     test_2018 = ISICDataset(
    #         dataset_root / config['data']['test_2018']['images'],
    #         dataset_root / config['data']['test_2018']['labels'],
    #         transform=transforms_val,
    #         year=2018
    #     )
    #     datasets['test'].append(test_2018)
    #     print(f"Loaded 2018 test dataset: {len(test_2018)} samples")
    # except Exception as e:
    #     print(f"Warning: Could not load 2018 dataset: {e}")
    #     print(f"Attempted paths:")
    #     print(f"Train: {dataset_root / config['data']['train_2018']['images']}")
    #     print(f"Labels: {dataset_root / config['data']['train_2018']['labels']}")
    
    # Verify we have data
    if not datasets['train']:
        raise ValueError("No training datasets could be loaded!")
    
    return datasets

def main():
    # Load configuration
    with open('src/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    log_dir = Path(config['logging']['log_dir'])
    save_dir = Path(config['logging']['save_dir'])
    log_dir.mkdir(exist_ok=True)
    save_dir.mkdir(exist_ok=True)
    
    logger = setup_logging(log_dir / 'training.log')
    writer = SummaryWriter(log_dir=log_dir)
    
    # Setup device
    device = (
        torch.device('mps') if torch.backends.mps.is_available() 
        else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    logger.info(f"Using device: {device}")
    
    # Create transforms
    transforms_train = get_transforms(config, is_train=True)
    transforms_val = get_transforms(config, is_train=False)
    
    # Create datasets
    datasets = create_datasets(config, transforms_train, transforms_val)
    
    # Create concatenated datasets
    train_dataset = ConcatDataset(datasets['train'])
    val_dataset = ConcatDataset(datasets['val'])
    test_dataset = ConcatDataset(datasets['test'])
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # Create model
    model = create_model(config).to(device)
    
    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config['training']['scheduler']['patience'],
        factor=config['training']['scheduler']['factor']
    )
    
    # Create metrics manager
    metrics_manager = MetricsManager(config)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            pred = output.argmax(dim=1).cpu().numpy()
            target = target.argmax(dim=1).cpu().numpy()
            train_preds.extend(pred)
            train_labels.extend(target)
        
        # Calculate training metrics
        train_metrics = metrics_manager.update_metrics(
            np.array(train_labels),
            np.array(train_preds),
            epoch,
            'train'
        )
        train_metrics['loss'] = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validation'):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                
                pred = output.argmax(dim=1).cpu().numpy()
                target = target.argmax(dim=1).cpu().numpy()
                val_preds.extend(pred)
                val_labels.extend(target)
        
        # Calculate validation metrics
        val_metrics = metrics_manager.update_metrics(
            np.array(val_labels),
            np.array(val_preds),
            epoch,
            'val'
        )
        val_metrics['loss'] = val_loss / len(val_loader)
        
        # Log metrics
        logger.info(f"Epoch {epoch} - Train: {train_metrics}")
        logger.info(f"Epoch {epoch} - Validation: {val_metrics}")
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= config['training']['early_stopping']['patience']:
            logger.info("Early stopping triggered")
            break
    
    # Final evaluation on test set
    model.eval()
    test_loss = 0
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            
            pred = output.argmax(dim=1).cpu().numpy()
            target = target.argmax(dim=1).cpu().numpy()
            test_preds.extend(pred)
            test_labels.extend(target)
    
    # Calculate and log test metrics
    test_metrics = metrics_manager.update_metrics(
        np.array(test_labels),
        np.array(test_preds),
        epoch,
        'test'
    )
    test_metrics['loss'] = test_loss / len(test_loader)
    logger.info(f"Test metrics: {test_metrics}")
    
    # Generate final plots and export results
    metrics_manager.plot_training_curves()
    metrics_manager.plot_confusion_matrix(
        np.array(test_labels),
        np.array(test_preds),
        config['classes']
    )
    metrics_manager.export_results()
    
    logger.info("Training completed!")
    writer.close()

if __name__ == '__main__':
    main()
