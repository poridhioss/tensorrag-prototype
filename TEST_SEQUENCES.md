# Test Sequences - Complete Collection

## Test Sequence 1: Basic CPU Training (Simplest)

```
Data Load → Data Split → Model Define → Train → Evaluate
```

**Configuration:**
- Data Load: `sample` → `california_housing`
- Data Split: `0.8` / `0.2`
- Model Define: `linear_regression`
- Train: Target = `MedHouseVal`
- Evaluate: Target = `MedHouseVal`

**Purpose:** Test basic CPU-based training pipeline

---

## Test Sequence 2: GPU Training (Full Card)

```
Data Load → Data Split → Model Define (GPU) → Train (GPU) → Evaluate
```

**Configuration:**
- Data Load: `sample` → `boston_housing`
- Data Split: `0.8` / `0.2`
- Model Define (GPU): 
  - Model Type: `deep_neural_network`
  - Hidden Layers: `[128, 64, 32]`
  - Activation: `relu`
  - Dropout: `0.2`
  - Learning Rate: `0.001`
  - Epochs: `20`
  - Batch Size: `32`
- Train (GPU): Target = `MEDV`
- Evaluate: Target = `MEDV`

**Purpose:** Test GPU training with full Train card

---

## Test Sequence 3: Multiple Model Types

```
Data Load → Data Split → Model Define → Train → Evaluate
         └─→ Model Define → Train → Evaluate
```

**Configuration:**
- Data Load: `sample` → `california_housing`
- Data Split: `0.8` / `0.2`
- **Branch 1:**
  - Model Define: `linear_regression`
  - Train: Target = `MedHouseVal`
  - Evaluate: Target = `MedHouseVal`
- **Branch 2:**
  - Model Define: `ridge` (alpha: `0.5`)
  - Train: Target = `MedHouseVal`
  - Evaluate: Target = `MedHouseVal`

**Purpose:** Test parallel model training and comparison

---

## Test Sequence 4: Individual Training Steps (One Batch)

```
Data Load → Data Split → Model Define (GPU) → Build Model → 
Initialize Optimizer → Prepare Batch → Zero Gradients → 
Forward Pass → Calculate Loss → Backward Pass → Optimizer Step
```

**Configuration:**
- Data Load: `sample` → `boston_housing`
- Data Split: `0.8` / `0.2`
- Model Define (GPU): `neural_network`, `[64, 32]`, `relu`, `0.2`, `0.001`, `5 epochs`, `32 batch`
- Build Model: `input_size: 13`
- Initialize Optimizer: `adam`, `0.001`
- Prepare Batch: `batch_index: 0`, `batch_size: 32`
- Calculate Loss: `mse`

**Purpose:** Test individual training step cards (one batch only)

---

## Test Sequence 5: Training Loop (Multiple Batches)

```
Data Load → Data Split → Model Define (GPU) → Build Model → 
Initialize Optimizer → 
[Batch 0: Prepare Batch → Zero Grad → Forward → Loss → Backward → Optimizer Step] →
[Batch 1: Prepare Batch → Zero Grad → Forward → Loss → Backward → Optimizer Step] →
[Batch 2: Prepare Batch → Zero Grad → Forward → Loss → Backward → Optimizer Step] →
Evaluate
```

**Configuration:**
- Same as Sequence 4, but create multiple Prepare Batch cards with different `batch_index` values
- Connect Optimizer Step outputs back to next batch's Zero Gradients and Forward Pass

**Purpose:** Test full training loop with multiple batches

---

## Test Sequence 6: Data Pipeline Only

```
Data Load → Data Split
```

**Configuration:**
- Data Load: `sample` → `california_housing`
- Data Split: `0.7` / `0.3`

**Purpose:** Test data loading and splitting, verify outputs

---

## Test Sequence 7: Model Definition Only

```
Model Define → Model Define (GPU)
```

**Configuration:**
- Model Define: `linear_regression`
- Model Define (GPU): `deep_neural_network`, `[64, 32]`, `relu`

**Purpose:** Test model definition cards, compare outputs

---

## Test Sequence 8: Inference Pipeline

```
Data Load → Data Split → Model Define → Train → Inference
```

**Configuration:**
- Data Load: `sample` → `california_housing`
- Data Split: `0.8` / `0.2`
- Model Define: `linear_regression`
- Train: Target = `MedHouseVal`
- Inference: 
  - Use test dataset or new data
  - Target = `MedHouseVal`

**Purpose:** Test inference on trained model

---

## Test Sequence 9: Full GPU Pipeline with Evaluation

```
Data Load → Data Split → Model Define (GPU) → Build Model → 
Train (GPU) → Evaluate → Inference
```

**Configuration:**
- Data Load: `sample` → `boston_housing`
- Data Split: `0.8` / `0.2`
- Model Define (GPU): `deep_neural_network`, `[128, 64]`, `relu`, `0.2`, `0.001`, `30 epochs`, `32 batch`
- Build Model: `input_size: 13`
- Train (GPU): Target = `MEDV`
- Evaluate: Target = `MEDV`
- Inference: Use test dataset

**Purpose:** Complete GPU workflow from data to inference

---

## Test Sequence 10: Training Steps with Evaluation

```
Data Load → Data Split → Model Define (GPU) → Build Model → 
Initialize Optimizer → Prepare Batch → Zero Grad → Forward → 
Loss → Backward → Optimizer Step → Evaluate
```

**Configuration:**
- Same as Sequence 4, but add Evaluate at the end
- Connect Optimizer Step `model` → Evaluate `trained_model`
- Connect Data Split `test_dataset` → Evaluate `test_dataset`

**Purpose:** Test individual steps with evaluation

---

## Test Sequence 11: Different Loss Functions

```
Data Load → Data Split → Model Define (GPU) → Build Model → 
Initialize Optimizer → Prepare Batch → Zero Grad → Forward → 
Calculate Loss (mse) → Backward → Optimizer Step
```

**Then test with:**
- Calculate Loss (mae)
- Calculate Loss (huber)

**Purpose:** Test different loss functions

---

## Test Sequence 12: Different Optimizers

```
Data Load → Data Split → Model Define (GPU) → Build Model → 
Initialize Optimizer (adam) → [Training Steps] → Evaluate
```

**Then test with:**
- Initialize Optimizer (sgd)
- Initialize Optimizer (rmsprop)

**Purpose:** Compare different optimizers

---

## Test Sequence 13: Gradient Clipping

```
Data Load → Data Split → Model Define (GPU) → Build Model → 
Initialize Optimizer → Prepare Batch → Zero Grad → Forward → 
Loss → Backward → Optimizer Step (gradient_clip: 1.0) → Evaluate
```

**Configuration:**
- Optimizer Step: `gradient_clip: 1.0` (clips gradients)

**Purpose:** Test gradient clipping feature

---

## Test Sequence 14: Multiple Epochs Simulation

```
Data Load → Data Split → Model Define (GPU) → Build Model → 
Initialize Optimizer → 
[Epoch 1: All batches] →
[Epoch 2: All batches] →
[Epoch 3: All batches] →
Evaluate
```

**Purpose:** Test multi-epoch training (manual setup)

---

## Test Sequence 15: Model Comparison

```
Data Load → Data Split → 
    ├─→ Model Define → Train → Evaluate (Model 1)
    └─→ Model Define (GPU) → Train (GPU) → Evaluate (Model 2)
```

**Purpose:** Compare CPU vs GPU model performance

---

## Test Sequence 16: Custom Dataset (CSV)

```
Data Load (csv) → Data Split → Model Define → Train → Evaluate
```

**Configuration:**
- Data Load: `source: csv`, `path: /path/to/your/data.csv`
- Rest same as Sequence 1

**Purpose:** Test with your own CSV file

---

## Test Sequence 17: Custom Dataset (URL)

```
Data Load (url) → Data Split → Model Define → Train → Evaluate
```

**Configuration:**
- Data Load: `source: url`, `url: https://example.com/data.csv`
- Rest same as Sequence 1

**Purpose:** Test loading data from URL

---

## Test Sequence 18: Complex Architecture

```
Data Load → Data Split → Model Define (GPU) → Train (GPU) → Evaluate
```

**Configuration:**
- Model Define (GPU):
  - Hidden Layers: `[256, 128, 64, 32, 16]` (deep network)
  - Activation: `tanh`
  - Dropout: `0.3`
  - Learning Rate: `0.0001`
  - Epochs: `100`

**Purpose:** Test complex neural network architecture

---

## Test Sequence 19: Minimal Training (Quick Test)

```
Data Load → Data Split → Model Define (GPU) → Train (GPU) → Evaluate
```

**Configuration:**
- Model Define (GPU):
  - Hidden Layers: `[32]` (simple)
  - Epochs: `2` (very few)
  - Batch Size: `64` (larger batches)

**Purpose:** Quick test with minimal training

---

## Test Sequence 20: Full Individual Steps (Complete)

```
Data Load → Data Split → Model Define (GPU) → Build Model → 
Initialize Optimizer → 
Prepare Batch (0) → Zero Grad → Forward → Loss → Backward → Optimizer Step →
Prepare Batch (1) → Zero Grad → Forward → Loss → Backward → Optimizer Step →
Prepare Batch (2) → Zero Grad → Forward → Loss → Backward → Optimizer Step →
Evaluate
```

**Purpose:** Complete manual training loop with multiple batches

---

## Recommended Test Order

### Beginner (Start Here)
1. **Sequence 1** - Basic CPU Training
2. **Sequence 6** - Data Pipeline Only
3. **Sequence 2** - GPU Training (Full Card)

### Intermediate
4. **Sequence 9** - Full GPU Pipeline
5. **Sequence 4** - Individual Steps (One Batch)
6. **Sequence 8** - Inference Pipeline

### Advanced
7. **Sequence 5** - Training Loop (Multiple Batches)
8. **Sequence 20** - Full Individual Steps
9. **Sequence 11** - Different Loss Functions
10. **Sequence 12** - Different Optimizers

### Custom Data
11. **Sequence 16** - Custom CSV
12. **Sequence 17** - Custom URL

---

## Quick Test Matrix

| Sequence | Complexity | Time | Purpose |
|----------|-----------|------|---------|
| 1 | ⭐ Easy | 1 min | Basic functionality |
| 2 | ⭐⭐ Medium | 3 min | GPU training |
| 4 | ⭐⭐⭐ Hard | 5 min | Individual steps |
| 5 | ⭐⭐⭐⭐ Very Hard | 10+ min | Full manual loop |
| 9 | ⭐⭐ Medium | 4 min | Complete workflow |

---

## Testing Tips

1. **Start Simple:** Begin with Sequence 1
2. **Verify Each Step:** Check outputs at each card
3. **Watch Logs:** Monitor backend terminal for execution
4. **Check Console:** Frontend console shows WebSocket updates
5. **Inspect Outputs:** Click "Output" on each completed card
6. **Test Incrementally:** Add complexity gradually

---

## Expected Results

### Sequence 1 (CPU)
- Execution time: ~10-30 seconds
- All cards: Green (completed)
- Evaluate shows: MSE, RMSE, R² metrics

### Sequence 2 (GPU)
- Execution time: ~2-5 minutes
- Train card runs on Modal GPU
- Evaluate shows neural network metrics

### Sequence 4 (Individual Steps)
- Execution time: ~1-2 minutes (one batch)
- Each step completes sequentially
- Model state preserved between steps

---

## Troubleshooting by Sequence

### If Sequence 1 Fails:
- Check backend is running
- Verify cards are loaded
- Check data loading works

### If Sequence 2 Fails:
- Verify Modal deployment
- Check GPU availability
- Verify model configuration

### If Sequence 4 Fails:
- Check all connections are correct
- Verify model flows through steps
- Check batch data connections

---

## Success Criteria

✅ **Sequence 1:** All cards complete, metrics displayed  
✅ **Sequence 2:** GPU training completes, model evaluated  
✅ **Sequence 4:** Each step executes, model updates  
✅ **Sequence 9:** Complete pipeline from data to inference  

---

## Next Steps After Testing

Once these sequences work:

1. Experiment with different architectures
2. Try your own datasets
3. Build custom training loops
4. Optimize hyperparameters
5. Compare different approaches
