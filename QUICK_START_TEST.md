# Quick Start - Testing the Full Setup

## Prerequisites

1. ✅ Backend dependencies installed
2. ✅ Frontend dependencies installed  
3. ✅ Modal app deployed
4. ✅ Both servers can run

---

## Quick Test (5 Minutes)

### Step 1: Start Backend

```bash
cd /home/fazlul/Downloads/tensorRag/backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

**Keep this terminal open!**

### Step 2: Start Frontend (New Terminal)

```bash
cd /home/fazlul/Downloads/tensorRag/frontend
npm run dev
```

**Keep this terminal open!**

### Step 3: Run Quick Test Script (Optional)

```bash
cd /home/fazlul/Downloads/tensorRag
./quick_test.sh
```

### Step 4: Open Browser

Go to: **http://localhost:3000**

---

## Test 1: Simple CPU Pipeline (2 minutes)

### Create This Pipeline:

```
Data Load → Data Split → Model Define → Train → Evaluate
```

### Configuration:

1. **Data Load Card:**
   - Source: `sample`
   - Sample Name: `california_housing`

2. **Data Split Card:**
   - Train Ratio: `0.8`
   - Test Ratio: `0.2`

3. **Model Define Card:**
   - Model Type: `linear_regression`

4. **Train Card:**
   - Target Column: `MedHouseVal`
   - Feature Columns: (leave empty - uses all)

5. **Evaluate Card:**
   - Target Column: `MedHouseVal`

### Connect:

- Data Load `dataset` → Data Split `dataset`
- Data Split `train_dataset` → Train `train_dataset`
- Data Split `test_dataset` → Evaluate `test_dataset`
- Model Define `model_spec` → Train `model_spec`
- Train `trained_model` → Evaluate `trained_model`

### Run:

- Click the "Run" button in header (or play icon on Data Load card)
- Watch cards turn green as they complete
- Check outputs by clicking "Output" on each card

**Expected:** All cards complete successfully, Evaluate shows metrics

---

## Test 2: GPU Training with Full Train Card (3 minutes)

### Create This Pipeline:

```
Data Load → Data Split → Model Define (GPU) → Train (GPU) → Evaluate
```

### Configuration:

1. **Data Load:**
   - Source: `sample`
   - Sample Name: `boston_housing`

2. **Data Split:**
   - Train Ratio: `0.8`
   - Test Ratio: `0.2`

3. **Model Define (GPU):**
   - Model Type: `deep_neural_network`
   - Hidden Layers: `[64, 32]`
   - Activation: `relu`
   - Dropout: `0.2`
   - Learning Rate: `0.001`
   - Epochs: `10` (use fewer for testing)
   - Batch Size: `32`

4. **Train (GPU):**
   - Target Column: `MEDV`

5. **Evaluate:**
   - Target Column: `MEDV`

### Connect:

- Data Load → Data Split
- Data Split `train_dataset` → Train (GPU)
- Data Split `test_dataset` → Evaluate
- Model Define (GPU) `model_spec` → Train (GPU)
- Train (GPU) `trained_model` → Evaluate

### Run:

- Click Run
- This will execute on Modal with GPU
- Takes longer than CPU training

**Expected:** GPU training completes, model is evaluated

---

## Test 3: Individual Training Steps (Advanced - 5 minutes)

### Create This Pipeline:

```
Data Load → Data Split → Model Define (GPU) → Build Model → 
Initialize Optimizer → Prepare Batch → Zero Gradients → 
Forward Pass → Calculate Loss → Backward Pass → Optimizer Step
```

### Configuration:

- **Data Load**: `sample` → `boston_housing`
- **Data Split**: `0.8` / `0.2`
- **Model Define (GPU)**: `deep_neural_network`, `[64, 32]`, `relu`, `0.2`, `0.001`, `5 epochs`, `32 batch`
- **Build Model**: `input_size: 13`
- **Initialize Optimizer**: `adam`, `0.001`
- **Prepare Batch**: `batch_index: 0`, `batch_size: 32`
- **Calculate Loss**: `mse`

### Connect (Follow SINGLE_CONNECTION_FLOW.md):

1. Data Load → Data Split
2. Model Define (GPU) → Build Model
3. Build Model → Initialize Optimizer
4. Initialize Optimizer `optimizer` → Zero Gradients
5. Initialize Optimizer `model` → Forward Pass
6. Data Split `train_dataset` → Prepare Batch
7. Prepare Batch `batch_data` → Forward Pass (edge 1)
8. Prepare Batch `batch_data` → Calculate Loss (edge 2)
9. Zero Gradients → Optimizer Step `optimizer`
10. Forward Pass `predictions` → Calculate Loss
11. Forward Pass `model` → Backward Pass
12. Calculate Loss → Backward Pass
13. Backward Pass → Optimizer Step `model`

### Run:

- Click Run
- Watch each step execute
- Check outputs at each stage

**Expected:** Each step completes, model gets updated

---

## Verification Checklist

After running tests, verify:

- [ ] Backend logs show execution
- [ ] Frontend shows card status updates
- [ ] WebSocket connection works (see console logs)
- [ ] Card outputs are displayed correctly
- [ ] No errors in browser console
- [ ] No errors in backend terminal
- [ ] Storage directory has pipeline outputs

---

## Troubleshooting

### Backend Not Starting
```bash
# Check if port is in use
lsof -i :8000

# Try different port
uvicorn app.main:app --reload --port 8001
# Then update frontend .env.local: NEXT_PUBLIC_API_URL=http://localhost:8001
```

### Frontend Can't Connect
```bash
# Check backend is running
curl http://localhost:8000/api/health

# Check CORS settings in backend/app/config.py
```

### Cards Not Loading
```bash
# Test cards endpoint
curl http://localhost:8000/api/cards | python3 -m json.tool
```

### Modal Errors
```bash
# Redeploy Modal app
cd backend
source .venv/bin/activate
modal deploy cards/modal_app.py
```

---

## Success Indicators

✅ **Backend:** Terminal shows "Application startup complete"  
✅ **Frontend:** Browser shows TensorRag interface with cards in sidebar  
✅ **Cards:** Can drag, connect, and configure cards  
✅ **Execution:** Cards complete and show green status  
✅ **Outputs:** Can view outputs by clicking "Output" button  
✅ **WebSocket:** Console shows real-time status updates  

---

## Next Steps

Once basic tests pass:

1. Try different datasets
2. Experiment with different model architectures
3. Test with your own data
4. Build more complex pipelines
5. Explore individual training steps for custom logic

---

## Quick Reference

**Backend:** http://localhost:8000  
**Frontend:** http://localhost:3000  
**Health Check:** http://localhost:8000/api/health  
**Cards API:** http://localhost:8000/api/cards  

**Start Commands:**
```bash
# Terminal 1: Backend
cd backend && source .venv/bin/activate && uvicorn app.main:app --reload

# Terminal 2: Frontend  
cd frontend && npm run dev
```
