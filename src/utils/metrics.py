import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def calculate_metrics(predictions, labels):
    """
    Calculate various classification metrics
    
    Args:
        predictions: numpy array of predicted labels
        labels: numpy array of true labels
        
    Returns:
        dict: Dictionary containing various metrics
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    
    # Ensure same shape
    assert predictions.shape == labels.shape, f"Shape mismatch: predictions {predictions.shape}, labels {labels.shape}"
    
    # Calculate basic metrics
    accuracy = accuracy_score(labels, predictions)
    
    # Calculate precision, recall, and f1 for each class
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        predictions, 
        average='weighted',
        zero_division=0
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }
    
    return metrics
