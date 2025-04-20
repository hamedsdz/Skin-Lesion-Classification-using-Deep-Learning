import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def plot_training_history(metrics_df, save_dir='results/plots'):
    """Plot training and validation metrics over time"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn')
    
    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics = ['accuracy', 'loss', 'precision', 'f1']
    
    for metric, ax in zip(metrics, axes.flat):
        # Plot training metrics
        train_data = metrics_df[metrics_df['phase'] == 'train']
        val_data = metrics_df[metrics_df['phase'] == 'val']
        
        ax.plot(train_data['epoch'], train_data[metric], 
                label=f'Training {metric}', marker='o')
        ax.plot(val_data['epoch'], val_data[metric], 
                label=f'Validation {metric}', marker='o')
        
        ax.set_title(f'{metric.capitalize()} over epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_dir='results/plots'):
    """Plot confusion matrix"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(y_true, y_pred_proba, class_names, save_dir='results/plots'):
    """Plot ROC curves for each class"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    for i in range(len(class_names)):
        fpr, tpr, _ = sklearn.metrics.roc_curve(
            (y_true == i).astype(int), 
            y_pred_proba[:, i]
        )
        auc = sklearn.metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def export_metrics(metrics_df, save_dir='results'):
    """Export metrics to CSV and generate summary report"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Export full metrics
    metrics_df.to_csv(save_dir / 'training_metrics.csv', index=False)
    
    # Generate summary report
    summary = {
        'train': metrics_df[metrics_df['phase'] == 'train'].describe(),
        'val': metrics_df[metrics_df['phase'] == 'val'].describe()
    }
    
    with open(save_dir / 'training_summary.txt', 'w') as f:
        f.write('Training Summary\n')
        f.write('================\n\n')
        
        for phase in ['train', 'val']:
            f.write(f'\n{phase.capitalize()} Metrics:\n')
            f.write('-' * 40 + '\n')
            f.write(summary[phase].to_string())
            f.write('\n\n')
        
        # Best performance
        best_val_acc = metrics_df[metrics_df['phase'] == 'val']['accuracy'].max()
        best_val_epoch = metrics_df[
            (metrics_df['phase'] == 'val') & 
            (metrics_df['accuracy'] == best_val_acc)
        ]['epoch'].iloc[0]
        
        f.write('\nBest Performance:\n')
        f.write('-' * 40 + '\n')
        f.write(f'Best validation accuracy: {best_val_acc:.4f} (epoch {best_val_epoch})\n')

def create_results_visualizations(metrics_df, test_results, class_names):
    """Create all visualizations and exports"""
    # Plot training history
    plot_training_history(metrics_df)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        test_results['true_labels'],
        test_results['predictions'],
        class_names
    )
    
    # Plot ROC curves
    plot_roc_curves(
        test_results['true_labels'],
        test_results['probabilities'],
        class_names
    )
    
    # Export metrics
    export_metrics(metrics_df)
