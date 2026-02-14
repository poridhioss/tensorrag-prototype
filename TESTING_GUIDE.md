# Testing Guide - Full Setup

## Prerequisites Check

### 1. Backend Dependencies
```bash
cd backend
source .venv/bin/activate  # or use uv
pip list | grep -E "fastapi|modal|pandas|torch"
```

### 2. Frontend Dependencies
```bash
cd frontend
npm list | grep -E "next|react|@xyflow"
```

### 3. Modal Deployment
```bash
cd backend
source .venv/bin/activate
modal app list
# Should show "tensorrag" app
```

---

## Step 1: Start Backend Server

### Terminal 1: Backend
```bash
cd /home/fazlul/Downloads/tensorRag/backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

**Test Backend:**
```bash
# In another terminal
curl http://localhost:8000/api/health
# Should return: {"status":"ok"}

curl http://localhost:8000/api/cards
# Should return list of cards
```

---

## Step 2: Start Frontend Server

### Terminal 2: Frontend
```bash
cd /home/fazlul/Downloads/tensorRag/frontend
npm run dev
```

**Expected Output:**
```
▲ Next.js 16.1.6
- Local:        http://localhost:3000
- Ready in XXXms
```

**Test Frontend:**
- Open browser: http://localhost:3000
- Should see the TensorRag interface
- Sidebar should show card categories

---

## Step 3: Verify Modal Deployment

### Check Modal App
```bash
cd backend
source .venv/bin/activate
modal app list
```

**Expected:**
- App name: `tensorrag`
- Functions: `run_card`, `run_card_gpu`

### Test Modal Function (Optional)
```bash
cd backend
source .venv/bin/activate
python -c "
import modal
func = modal.Function.from_name('tensorrag', 'run_card')
print('Modal function found:', func)
"
```

---

## Step 4: Test Simple Pipeline (CPU Model)

### Test Case 1: Simple CPU Training

**Pipeline:**
```
Data Load → Data Split → Model Define → Train → Evaluate
```

**Steps:**

1. **Open Frontend**: http://localhost:3000

2. **Add Cards:**
   - Drag "Data Load" card
   - Drag "Data Split" card
   - Drag "Model Define" card
   - Drag "Train" card
   - Drag "Evaluate" card

3. **Configure Cards:**
   - **Data Load**: 
     - Source: `sample`
     - Sample Name: `california_housing`
   - **Data Split**:
     - Train Ratio: `0.8`
     - Test Ratio: `0.2`
   - **Model Define**:
     - Model Type: `linear_regression`
   - **Train**:
     - Target Column: `MedHouseVal`
   - **Evaluate**:
     - Target Column: `MedHouseVal`

4. **Connect Cards:**
   - Data Load `dataset` → Data Split `dataset`
   - Data Split `train_dataset` → Train `train_dataset`
   - Data Split `test_dataset` → Evaluate `test_dataset`
   - Model Define `model_spec` → Train `model_spec`
   - Train `trained_model` → Evaluate `trained_model`

5. **Run Pipeline:**
   - Click "Run" button (or play icon on Data Load card)
   - Watch execution in console/logs
   - Check card status indicators

**Expected Results:**
- All cards should turn green (completed)
- Evaluate card should show metrics (MSE, R²)
- Click "Output" on cards to see results

---

## Step 5: Test GPU Pipeline (Full Training Steps)

### Test Case 2: GPU Training with Individual Steps

**Pipeline:**
```
Data Load → Data Split → Model Define (GPU) → Build Model → 
Initialize Optimizer → Prepare Batch → Zero Grad → Forward Pass → 
Calculate Loss → Backward Pass → Optimizer Step → Evaluate
```

**Steps:**

1. **Add Cards** (in order):
   - Data Load
   - Data Split
   - Model Define (GPU)
   - Build Model
   - Initialize Optimizer
   - Prepare Batch
   - Zero Gradients
   - Forward Pass
   - Calculate Loss
   - Backward Pass
   - Optimizer Step
   - Evaluate

2. **Configure:**
   - **Data Load**: `sample` → `boston_housing`
   - **Data Split**: `0.8` / `0.2`
   - **Model Define (GPU)**:
     - Model Type: `deep_neural_network`
     - Hidden Layers: `[64, 32]`
     - Activation: `relu`
     - Dropout: `0.2`
     - Learning Rate: `0.001`
     - Epochs: `10` (for testing, use fewer)
     - Batch Size: `32`
   - **Build Model**:
     - Input Size: `13` (boston_housing has 13 features)
   - **Initialize Optimizer**:
     - Optimizer Type: `adam`
     - Learning Rate: `0.001`
   - **Prepare Batch**:
     - Batch Index: `0`
     - Batch Size: `32`
   - **Calculate Loss**:
     - Loss Type: `mse`

3. **Connect Cards:**
   ```
   Data Load → Data Split
   Data Split → Prepare Batch (train_dataset)
   Data Split → Evaluate (test_dataset)
   
   Model Define (GPU) → Build Model
   Build Model → Initialize Optimizer
   Initialize Optimizer → Zero Gradients (optimizer)
   Initialize Optimizer → Forward Pass (model)
   
   Prepare Batch → Forward Pass (batch_data)
   Prepare Batch → Calculate Loss (batch_data)
   
   Zero Gradients → Optimizer Step (optimizer)
   Forward Pass → Calculate Loss (predictions)
   Forward Pass → Backward Pass (model)
   Calculate Loss → Backward Pass (loss)
   Backward Pass → Optimizer Step (model)
   
   Optimizer Step → Evaluate (model)
   ```

4. **Run Pipeline:**
   - Click Run button
   - Monitor execution
   - Check for errors

**Expected Results:**
- Cards execute in order
- GPU training runs on Modal
- Final model available for evaluation

---

## Step 6: Test GPU Training (Full Train Card)

### Test Case 3: Simplified GPU Training

**Pipeline:**
```
Data Load → Data Split → Model Define (GPU) → Train (GPU) → Evaluate
```

**Steps:**

1. **Add Cards:**
   - Data Load
   - Data Split
   - Model Define (GPU)
   - Train (GPU)
   - Evaluate

2. **Configure:**
   - **Data Load**: `sample` → `boston_housing`
   - **Data Split**: `0.8` / `0.2`
   - **Model Define (GPU)**: Same as above
   - **Train (GPU)**:
     - Target Column: `MEDV`
   - **Evaluate**:
     - Target Column: `MEDV`

3. **Connect:**
   - Data Load → Data Split
   - Data Split → Train (GPU) (train_dataset)
   - Data Split → Evaluate (test_dataset)
   - Model Define (GPU) → Train (GPU) (model_spec)
   - Train (GPU) → Evaluate (trained_model)

4. **Run:**
   - Click Run
   - Should complete automatically

**Expected:**
- Faster execution (single card handles training loop)
- All cards complete successfully
- Metrics displayed in Evaluate card

---

## Step 7: Verify Outputs

### Check Card Outputs

1. **Data Load Output:**
   - Click "Output" button on Data Load card
   - Should show data table with rows/columns

2. **Data Split Output:**
   - Click "Output" on Data Split card
   - Should show train/test row counts

3. **Train Output:**
   - Click "Output" on Train card
   - Should show training metrics (MSE, RMSE, R²)

4. **Evaluate Output:**
   - Click "Output" on Evaluate card
   - Should show evaluation metrics

### Check Backend Storage

```bash
cd backend
ls -la storage/
# Should see pipeline directories with outputs
```

---

## Step 8: Test Error Handling

### Test Invalid Connection
1. Try connecting incompatible types
2. Should be prevented by validation

### Test Missing Configuration
1. Try running without configuring required fields
2. Should show validation errors

### Test Modal Failure
1. Disconnect from Modal (if possible)
2. Try running GPU card
3. Should show appropriate error

---

## Common Issues & Solutions

### Issue 1: Backend Not Starting
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill process if needed
kill -9 <PID>

# Or use different port
uvicorn app.main:app --reload --port 8001
```

### Issue 2: Frontend Can't Connect to Backend
```bash
# Check backend is running
curl http://localhost:8000/api/health

# Check frontend .env.local
cat frontend/.env.local
# Should have: NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Issue 3: Modal Function Not Found
```bash
# Redeploy Modal app
cd backend
source .venv/bin/activate
modal deploy cards/modal_app.py
```

### Issue 4: Cards Not Loading
```bash
# Check backend cards endpoint
curl http://localhost:8000/api/cards | jq '.[].card_type'

# Should return list of card types
```

### Issue 5: Training Fails
- Check console logs in frontend
- Check backend terminal for errors
- Verify Modal deployment
- Check data format matches expectations

---

## Quick Test Script

### Automated Test (Backend)
```bash
cd backend
source .venv/bin/activate

# Test 1: Health check
curl http://localhost:8000/api/health

# Test 2: Get cards
curl http://localhost:8000/api/cards | python -m json.tool

# Test 3: Validate simple pipeline
python << EOF
import requests
import json

pipeline = {
    "pipeline_id": "test-001",
    "nodes": [
        {"id": "n1", "type": "data_load", "config": {"source": "sample", "sample_name": "california_housing"}},
        {"id": "n2", "type": "data_split", "config": {"train_ratio": 0.8, "test_ratio": 0.2}},
    ],
    "edges": [
        {"source": "n1", "target": "n2", "source_output": "dataset", "target_input": "dataset"}
    ]
}

response = requests.post("http://localhost:8000/api/pipeline/validate", json=pipeline)
print("Validation:", response.json())
EOF
```

---

## Testing Checklist

### Backend
- [ ] Backend server starts on port 8000
- [ ] Health endpoint returns OK
- [ ] Cards endpoint returns card list
- [ ] Modal app is deployed
- [ ] Storage directory exists

### Frontend
- [ ] Frontend starts on port 3000
- [ ] Cards load in sidebar
- [ ] Can drag cards to canvas
- [ ] Can connect cards
- [ ] Can configure cards
- [ ] Can run pipeline

### Pipeline Execution
- [ ] Simple CPU pipeline works
- [ ] GPU pipeline works (full Train card)
- [ ] Individual training steps work
- [ ] Outputs are displayed correctly
- [ ] Errors are shown properly

### Modal Integration
- [ ] GPU cards execute on Modal
- [ ] CPU cards execute locally
- [ ] Model state persists between cards
- [ ] Serialization/deserialization works

---

## Performance Testing

### Test Execution Times
1. **CPU Training** (simple model): Should complete in seconds
2. **GPU Training** (full card): Should complete in minutes
3. **Individual Steps**: Will be slower (multiple Modal calls)

### Monitor Resources
```bash
# Watch backend logs
tail -f backend/logs/*.log

# Monitor Modal usage
modal app logs tensorrag
```

---

## Next Steps After Testing

1. **If everything works:**
   - Try more complex pipelines
   - Experiment with different architectures
   - Test with your own datasets

2. **If issues found:**
   - Check error messages
   - Review logs
   - Verify configurations
   - Check Modal deployment status

---

## Getting Help

### Debug Mode
```bash
# Backend with verbose logging
uvicorn app.main:app --reload --log-level debug

# Frontend with React DevTools
# Install React DevTools browser extension
```

### Check Logs
- Backend: Terminal output
- Frontend: Browser console (F12)
- Modal: `modal app logs tensorrag`

### Common Debug Commands
```bash
# Check Modal functions
modal function list tensorrag

# Check Modal app status
modal app show tensorrag

# Test card registration
cd backend
python -c "from cards.registry import list_cards; print([c.card_type for c in list_cards()])"
```
