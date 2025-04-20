import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc
)
from datetime import datetime
import seaborn as sns
import os
from pathlib import Path

class MetricsManager:
    def __init__(self, config):
        self.config = config
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize metrics storage
        self.epoch_metrics = []
        self.best_metrics = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'epoch': 0
        }
        
        # Create unique experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"experiment_{timestamp}"
        
    def update_metrics(self, y_true, y_pred, epoch, phase='train'):
        """Calculate and store comprehensive metrics"""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Store metrics
        metrics = {
            'epoch': epoch,
            'phase': phase,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        self.epoch_metrics.append(metrics)
        
        # Update best metrics if validation phase
        if phase == 'val' and accuracy > self.best_metrics['accuracy']:
            self.best_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'epoch': epoch
            }
        
        return metrics
    
    def plot_training_curves(self):
        """Plot training and validation metrics over epochs"""
        metrics_df = pd.DataFrame(self.epoch_metrics)
        
        # Create plots directory
        plots_dir = self.results_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Plot metrics
        plt.figure(figsize=(15, 10))
        
        # Plot each metric
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 2, i)
            
            # Plot training and validation curves
            train_data = metrics_df[metrics_df['phase'] == 'train']
            val_data = metrics_df[metrics_df['phase'] == 'val']
            
            plt.plot(train_data['epoch'], train_data[metric], label=f'Train {metric}')
            plt.plot(val_data['epoch'], val_data[metric], label=f'Val {metric}')
            
            plt.title(f'{metric.capitalize()} over epochs')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'{self.experiment_name}_metrics.png')
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, classes):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plots_dir = self.results_dir / 'plots'
        plt.savefig(plots_dir / f'{self.experiment_name}_confusion_matrix.png')
        plt.close()
    
    def export_results(self):
        """Export all results to CSV"""
        # Export epoch metrics
        metrics_df = pd.DataFrame(self.epoch_metrics)
        metrics_df.to_csv(self.results_dir / f'{self.experiment_name}_metrics.csv', index=False)
        
        # Export best metrics
        best_metrics_df = pd.DataFrame([self.best_metrics])
        best_metrics_df.to_csv(self.results_dir / f'{self.experiment_name}_best_metrics.csv', index=False)
        
        # Create summary report
        with open(self.results_dir / f'{self.experiment_name}_summary.txt', 'w') as f:
            f.write("Training Summary\n")
            f.write("===============\n\n")
            f.write(f"Experiment Name: {self.experiment_name}\n")
            f.write(f"Best Epoch: {self.best_metrics['epoch']}\n")
            f.write("\nBest Metrics:\n")
            for metric, value in self.best_metrics.items():
                if metric != 'epoch':
                    f.write(f"{metric}: {value:.4f}\n")
