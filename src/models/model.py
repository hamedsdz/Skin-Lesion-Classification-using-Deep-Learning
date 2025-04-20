import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AttentionBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, in_features // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // 16, in_features, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return x * self.channel_att(x)

class SkinLesionClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Use ResNet18 as backbone for testing
        self.backbone = models.resnet18(pretrained=False)
        
        # Get the number of features from the backbone
        self.num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Attention mechanism
        self.attention = AttentionBlock(self.num_features)
        
        # Global pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Classifier
        self.dropout = nn.Dropout(p=config['model']['dropout_rate'])
        self.fc = nn.Linear(self.num_features * 2, config['model']['num_classes'])

    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        features = features.view(features.size(0), self.num_features, 1, 1)
        
        # Apply attention
        attended = self.attention(features)
        
        # Global pooling
        avg_pool = self.avg_pool(attended).flatten(1)
        max_pool = self.max_pool(attended).flatten(1)
        
        # Concatenate pooling results
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        
        # Apply dropout
        pooled = self.dropout(pooled)
        
        # Classification
        output = self.fc(pooled)
        
        return output

def create_model(config):
    """Factory function to create a model instance"""
    model = SkinLesionClassifier(config)
    return model
