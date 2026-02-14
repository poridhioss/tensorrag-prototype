# GPU-Based Training Guide

## Overview

TensorRag now supports GPU-based neural network training using PyTorch on Modal's serverless GPU infrastructure. This guide explains the new features and how to use them.

## What's New

### 1. New Dataset: Boston Housing (GPU-Optimized)
- **Dataset Name**: `boston_housing`
- **Size**: 50,000 samples (synthetic, larger dataset suitable for GPU training)
- **Features**: 13 features (CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT)
- **Target**: MEDV (Median value)
- **Usage**: Select "sample" as source and "boston_housing" as sample_name in Data Load card

### 2. New Cards

#### Model Define (GPU) Card
- **Card Type**: `model_define_gpu`
- **Purpose**: Define neural network architectures for GPU training
- **Configuration Options**:
  - `model_type`: "neural_network" or "deep_neural_network"
  - `hidden_layers`: Array of integers (e.g., [64, 32] for two hidden layers)
  - `activation`: "relu", "tanh", or "sigmoid"
  - `dropout`: Dropout rate (0-1)
  - `learning_rate`: Learning rate (default: 0.001)
  - `epochs`: Number of training epochs (default: 50)
  - `batch_size`: Batch size (default: 32)

#### Train (GPU) Card
- **Card Type**: `train_gpu`
- **Purpose**: Train neural networks on GPU using PyTorch
- **Execution**: Runs on Modal with T4 GPU
- **Features**:
  - Automatic feature normalization
  - GPU acceleration (falls back to CPU if GPU unavailable)
  - Training loss tracking
  - Comprehensive metrics (MSE, RMSE, R²)

## Usage Workflow

### Step 1: Load Dataset
1. Add a **Data Load** card
2. Configure:
   - Source: `sample`
   - Sample Name: `boston_housing`
3. Run the card to load the dataset

### Step 2: Split Data
1. Add a **Data Split** card
2. Connect it to the Data Load card
3. Configure train/test ratios (e.g., 80/20)
4. Run to split the dataset

### Step 3: Define GPU Model
1. Add a **Model Define (GPU)** card
2. Configure neural network:
   - Model Type: `neural_network` or `deep_neural_network`
   - Hidden Layers: `[128, 64, 32]` (example)
   - Activation: `relu`
   - Dropout: `0.2`
   - Learning Rate: `0.001`
   - Epochs: `50`
   - Batch Size: `32`

### Step 4: Train on GPU
1. Add a **Train (GPU)** card
2. Connect:
   - `train_dataset` from Data Split card
   - `model_spec` from Model Define (GPU) card
3. Configure:
   - Target Column: `MEDV`
   - Feature Columns: (leave empty to use all except target)
4. Run the card - it will execute on Modal with GPU

### Step 5: Evaluate
1. Add an **Evaluate** card
2. Connect:
   - `test_dataset` from Data Split card
   - `trained_model` from Train (GPU) card
3. Run to get evaluation metrics

## Architecture Details

### Modal GPU Function
- **Function Name**: `run_card_gpu`
- **GPU Type**: T4
- **Timeout**: 600 seconds (10 minutes)
- **Image**: Includes PyTorch with CUDA 11.8 support

### Neural Network Architecture
The GPU training card implements a feedforward neural network:
- Input layer: Size matches number of features
- Hidden layers: Configurable (e.g., [128, 64, 32])
- Activation: ReLU, Tanh, or Sigmoid
- Dropout: Applied after each hidden layer
- Output layer: Single neuron (regression)

### Training Process
1. **Data Preprocessing**:
   - Feature normalization (zero mean, unit variance)
   - Target normalization
2. **Model Training**:
   - Adam optimizer
   - MSE loss function
   - Configurable epochs and batch size
3. **Evaluation**:
   - Metrics calculated on denormalized predictions
   - Loss history tracked per epoch

## Output Preview

The Train (GPU) card provides:
- **Metrics**:
  - Train MSE, RMSE, R²
  - Final training loss
  - Loss history (last 10 epochs)
- **Metadata**:
  - Device used (cuda/cpu)
  - Number of epochs trained
  - Feature columns used

## Comparison: CPU vs GPU Training

| Feature | CPU (Train) | GPU (Train GPU) |
|---------|-------------|-----------------|
| Framework | scikit-learn | PyTorch |
| Models | Linear, Ridge, Lasso | Neural Networks |
| Execution | Local/Modal CPU | Modal GPU (T4) |
| Dataset Size | Small-Medium | Medium-Large |
| Training Time | Fast (seconds) | Moderate (minutes) |
| Use Case | Simple models | Complex patterns |

## Technical Notes

### GPU Availability
- The card automatically detects GPU availability
- Falls back to CPU if GPU is not available
- Output preview shows which device was used

### Model Serialization
- Models are moved to CPU before saving (for compatibility)
- Saved using joblib format
- Normalization parameters saved separately for inference

### Memory Considerations
- GPU memory is managed by Modal
- Batch size can be adjusted for memory constraints
- Larger datasets benefit more from GPU acceleration

## Example Pipeline

```
Data Load (boston_housing)
    ↓
Data Split (80/20)
    ↓
Model Define (GPU) [128, 64, 32] + Train (GPU)
    ↓
Evaluate
```

## Troubleshooting

### GPU Not Available
- Check Modal deployment status
- Verify `run_card_gpu` function is deployed
- Check card execution_mode is "modal"

### Training Fails
- Reduce batch size if out of memory
- Check feature columns match dataset
- Verify target column exists

### Slow Training
- Increase batch size (if memory allows)
- Reduce number of epochs for testing
- Check dataset size (GPU benefits larger datasets)

## Future Enhancements

Potential improvements:
- Support for more neural network architectures (CNN, RNN)
- Hyperparameter tuning
- Early stopping
- Learning rate scheduling
- Model checkpointing
- Multi-GPU training
