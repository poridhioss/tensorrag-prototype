# V1 Scope & Roadmap

## V1 Goal

Build a working visual pipeline that can train a simple linear regression model end-to-end: load data, split it, train, evaluate, and run inference — all through drag-and-drop cards. Lightweight cards (data loading, splitting, evaluation) run locally on the backend; heavy compute cards (training, inference) run on Modal.

---

## V1 Cards

Six cards ship in V1, covering a complete ML workflow:

### 1. Data Load Card
- **Input:** None (source node)
- **Config:** Source type (CSV upload, URL, sample dataset), column selection
- **Output:** Raw dataset (Parquet)
- **Output Viewer:** Data table (first 100 rows), column names & types, shape (rows x cols), basic stats (nulls, unique counts)
- **Runs:** Local (backend) — file I/O, no heavy compute

### 2. Data Split Card
- **Input:** Dataset (Parquet)
- **Config:** Train/test ratio, random seed, stratify column (optional)
- **Output:** Train set + Test set (Parquet)
- **Output Viewer:** Split summary (train rows, test rows, ratio), data table toggle between train/test sets
- **Runs:** Local (backend) — pandas operation, milliseconds

### 3. Model Define Card
- **Input:** None (config-only node)
- **Config:** Model type (Linear Regression, Ridge, Lasso for V1), hyperparameters
- **Output:** Model specification (JSON)
- **Output Viewer:** Model spec summary (type, hyperparameters as key-value list)
- **Runs:** Local (backend) — config-only, no computation

### 4. Train Card
- **Input:** Train dataset (Parquet) + Model specification (JSON)
- **Config:** Epochs (if iterative), target column, feature columns
- **Output:** Trained model (Joblib) + Training metrics (JSON)
- **Output Viewer:** Training metrics (final loss, convergence), model parameter summary (coefficients for linear models), loss curve chart (if iterative)
- **Runs:** Modal — may need GPU, long-running for larger models

### 5. Evaluate Card
- **Input:** Trained model (Joblib) + Test dataset (Parquet)
- **Config:** Metrics to compute (MSE, MAE, R2, etc.)
- **Output:** Evaluation report (JSON) + Visualizations (PNG)
- **Output Viewer:** Metric values as highlighted cards (e.g., R2: 0.94), residual plot, predicted vs actual scatter chart
- **Runs:** Local (backend) — lightweight metrics computation on test set

### 6. Inference Card
- **Input:** Trained model (Joblib) + New data (Parquet/CSV)
- **Config:** Output format, columns to include
- **Output:** Predictions (Parquet/CSV)
- **Output Viewer:** Predictions table (input features + predicted column side by side), row count, download button
- **Runs:** Modal — may need GPU for large models, batch predictions

---

## Implementation Phases

### Phase 1: Foundation
**Goal:** Scaffolding for both projects, basic board with static cards

Frontend (`frontend/`):
- [ ] Initialize Next.js project with TypeScript in `frontend/`
- [ ] Install and configure React Flow
- [ ] Set up Zustand store for pipeline state
- [ ] Set up Tailwind CSS with `darkMode: "class"` + next-themes
- [ ] Theme toggle component (light / dark / system) in header
- [ ] Define CSS variables for design tokens (colors, borders, shadows)
- [ ] Create basic board layout (sidebar + canvas)
- [ ] Build static card components (no execution, just drag-and-drop)
- [ ] Implement card connections with basic validation
- [ ] DAG serialization to JSON

Backend (`backend/`):
- [ ] Initialize FastAPI project in `backend/`
- [ ] Set up project structure (routers, services, models, cards)
- [ ] Configure CORS for frontend origin
- [ ] Health check endpoint (`GET /api/health`)

**Deliverable:** Two independent projects. Frontend: users can drag cards, connect them, and export pipeline JSON. Backend: running FastAPI server with CORS ready.

### Phase 2: Backend + Modal Integration
**Goal:** Pipeline execution engine (backend-only work)

- [ ] Set up Modal account and project
- [ ] Implement card interface contract (`cards/base.py`)
- [ ] Build Modal functions for each V1 card
- [ ] Implement DAG validator and topological sort (`services/dag.py`)
- [ ] Build execution engine — sequential for V1 (`services/executor.py`)
- [ ] Set up Modal Volumes for data transport (`services/storage.py`)
- [ ] API endpoints: validate, execute, status, card output
- [ ] Unit tests for DAG validation and executor

**Deliverable:** Backend can receive a pipeline JSON and execute it on Modal. Testable independently via curl/Postman.

### Phase 3: Connect Frontend to Backend
**Goal:** Wire the two projects together for end-to-end execution

- [ ] API client in frontend (`lib/api.ts`) pointing to `NEXT_PUBLIC_API_URL`
- [ ] "Run Pipeline" button — serializes DAG, POSTs to backend
- [ ] WebSocket connection for live status updates
- [ ] Card status indicators (pending/running/success/fail)
- [ ] Error display per card
- [ ] Basic log viewer
- [ ] Verify CORS and WebSocket work across separate origins

**Deliverable:** Users can build and run a pipeline from the UI and see live status. Frontend and backend communicate cleanly as separate services.

### Phase 4: Card Configuration
**Goal:** Per-card settings UI

- [ ] Side panel component for card configuration
- [ ] Config forms for each V1 card type
- [ ] Form validation against card config schemas
- [ ] Config persistence in pipeline state
- [ ] Per-card output viewer component (click card → see output)
- [ ] DataTableView — scrollable table for data cards (Load, Split, Inference)
- [ ] MetricsView — metric cards + charts for Evaluate card
- [ ] TrainLogView — loss curves, model summary for Train card
- [ ] Inline output summary on card node (compact: e.g., "1000 rows x 5 cols" or "R2: 0.94")
- [ ] Expand output to side panel for full-detail view

**Deliverable:** Users can configure each card and inspect output at every stage of the pipeline.

### Phase 5: Polish & Demo Pipeline
**Goal:** Demo-ready product

- [ ] Pre-built template: "Linear Regression on Housing Data"
- [ ] Onboarding flow / empty state
- [ ] Error handling and retry logic
- [ ] Loading states and animations
- [ ] Pipeline save/load (local storage for V1)
- [ ] Basic responsive layout

**Deliverable:** A polished demo showing end-to-end linear model training via visual pipeline.

---

## Post-V1 Considerations

Things explicitly **out of scope** for V1 but worth designing for:

| Feature | Why Later |
|---------|-----------|
| Deep Learning cards (PyTorch, TF) | Requires GPU config UI, longer training, checkpointing |
| Branching DAGs (ensemble, A/B eval) | V1 is linear pipelines only |
| User authentication | Not needed for single-user V1 |
| Pipeline versioning | Needs design for diffing, rollback |
| Custom card creation | Users write their own cards — big feature |
| Collaborative editing | Multi-user on same board |
| Scheduling / cron runs | Needs job queue infrastructure |
| Dataset management | Upload, version, browse datasets |
| Model registry | Store, version, deploy trained models |

---

## Complexity Estimate

| Phase | Complexity | Key Risk |
|-------|------------|----------|
| Phase 1: Foundation | Medium | React Flow learning curve, card type system design |
| Phase 2: Backend + Modal | High | Data transport between cards, Modal SDK integration |
| Phase 3: Connect FE-BE | Medium | WebSocket reliability, state sync |
| Phase 4: Card Config + Output | Medium-High | Form generation from schemas, output viewer rendering (tables, charts) |
| Phase 5: Polish | Low-Medium | Template pipeline, UX edge cases |

**Highest risk area:** Phase 2 — getting data to reliably flow between Modal functions via the object store. This should be prototyped early.

---

## Success Criteria for V1

- [ ] User can drag 6 card types onto the board
- [ ] User can connect cards to form a valid pipeline
- [ ] User can configure each card via the side panel
- [ ] User can click "Run" and see cards execute sequentially
- [ ] Live status updates show which card is running
- [ ] After execution, user can click any card to view its output (data table, metrics, charts)
- [ ] Each card shows a compact inline output summary on the board (shape, key metric, status)
- [ ] Demo pipeline (housing data linear regression) works end-to-end
