import os
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score
import yaml
import json

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.model import create_model
from src.data.dataset import ISICDataset, get_transforms
from src.utils.visualization import create_results_visualizations

def load_model(config, model_path):
    """Load trained model"""
    model = create_model(config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict(model, dataloader, device):
    """Make predictions on the dataset"""
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc='Testing'):
            data = data.to(device)
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(target.argmax(dim=1).cpu().numpy())
    
    return {
        'predictions': np.array(all_predictions),
        'probabilities': np.array(all_probabilities),
        'true_labels': np.array(all_labels)
    }

def evaluate(config, model_path, results_dir='results'):
    """Evaluate model on test set and save results"""
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # Create test dataset
    test_transform = get_transforms(config, is_train=False)
    
    # Load 2017 test dataset
    test_dataset_2017 = ISICDataset(
        config['data']['test_2017']['images'],
        config['data']['test_2017']['labels'],
        transform=test_transform,
        year=2017
    )
    
    # Load 2018 test dataset
    test_dataset_2018 = ISICDataset(
        config['data']['test_2018']['images'],
        config['data']['test_2018']['labels'],
        transform=test_transform,
        year=2018
    )
    
    # Combine datasets
    test_datasets = [test_dataset_2017, test_dataset_2018]
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(test_datasets),
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(config, model_path)
    model = model.to(device)
    
    # Make predictions
    results = predict(model, test_loader, device)
    
    # Calculate metrics
    class_names = config['classes']
    report = classification_report(
        results['true_labels'],
        results['predictions'],
        target_names=class_names,
        output_dict=True
    )
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(
        results['true_labels'],
        results['probabilities'],
        multi_class='ovr',
        average='weighted'
    )
    
    # Save results
    results_dict = {
        'classification_report': report,
        'roc_auc_score': roc_auc,
        'confusion_matrix': results
    }
    
    # Save detailed metrics
    with open(results_dir / 'test_metrics.json', 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    # Save predictions
    pd.DataFrame({
        'true_label': results['true_labels'],
        'predicted_label': results['predictions'],
        **{f'prob_{class_name}': results['probabilities'][:, i] 
           for i, class_name in enumerate(class_names)}
    }).to_csv(results_dir / 'predictions.csv', index=False)
    
    # Create visualizations
    create_results_visualizations(
        pd.read_csv(results_dir / 'training_metrics.csv'),
        results,
        class_names
    )
    
    return results_dict

if __name__ == '__main__':
    # Load configuration
    with open('src/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Evaluate model
    model_path = 'models/best_model.pth'
    results = evaluate(config, model_path)
    
    # Print summary
    print("\nTest Results:")
    print("=" * 50)
    print(f"Weighted ROC AUC: {results['roc_auc_score']:.4f}")
    print("\nPer-class metrics:")
    for class_name in config['classes']:
        metrics = results['classification_report'][class_name]
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1-score']:.4f}")
