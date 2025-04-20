# Skin Lesion Classification using Deep Learning

This project implements a deep learning solution for skin lesion classification using the ISIC (International Skin Imaging Collaboration) dataset. The model uses EfficientNet as a backbone with custom attention mechanisms for improved performance.

## Project Structure

```
.
├── dataset/          # ISIC dataset
├── src/             # Source code
│   ├── config/      # Configuration files
│   ├── data/        # Data processing
│   ├── models/      # Model architecture
│   ├── train/       # Training scripts
│   ├── evaluate/    # Evaluation scripts
│   └── utils/       # Utility functions
├── tests/           # Unit tests
├── logs/            # Training logs
├── models/          # Saved models
└── requirements/    # Project dependencies
```

## Dataset

The ISIC dataset can be downloaded from the official ISIC Archive: [https://www.isic-archive.com/](https://www.isic-archive.com/). 

To download the dataset:
1. Navigate to the ISIC Archive and create an account if you don't already have one.
2. Browse the available datasets and select the one required for your project (e.g., ISIC 2019 Challenge Dataset).
3. Download the dataset as a ZIP file and extract it into the `dataset/` directory.

Ensure the dataset structure matches the expected format specified in the `src/config/config.yaml` file.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements/dev.txt
   ```

3. Prepare the dataset:
   - The ISIC dataset should be organized in the dataset/ directory
   - Ensure the zip files are properly placed as specified in config.yaml

## Training

To train the model:

```bash
python src/train/train.py
```

The training script will:
- Load the configuration from src/config/config.yaml
- Process the ISIC dataset
- Train the model with the specified parameters
- Save checkpoints and logs
- Monitor training with TensorBoard

## Monitoring

Launch TensorBoard to monitor training:

```bash
tensorboard --logdir logs
```

## Model Architecture

The model uses:
- EfficientNet-B3 backbone
- Custom attention mechanisms
- Dual pooling strategy
- Advanced data augmentation

## Results

Training metrics and model checkpoints are saved in:
- logs/ directory for TensorBoard logs
- models/ directory for model checkpoints
- 
