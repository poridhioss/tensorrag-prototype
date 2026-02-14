# TensorRag

A visual DAG-based ML pipeline builder. Drag-and-drop independent card components, connect them, and execute ML workflows — from data loading to inference — with serverless compute on Modal.

1. Dragging **cards** (independent pipeline components) onto a board
2. Connecting cards to define data flow
3. Configuring each card via its UI
4. Executing the pipeline — each card runs on Modal's serverless infrastructure
5. Viewing **output at each card** — inspect intermediate results, data shapes, metrics, and visualizations inline

## Key Principles

- **Card Independence** — Every card is a self-contained unit with a standard interface. Cards are reusable across different pipelines (training, inference, fine-tuning, etc.)
- **Visual-First** — The React Flow board is the primary interface. No YAML, no config files — just drag, connect, and run. Light and dark mode included.
- **Serverless Execution** — Each card's computation runs on Modal, with automatic GPU allocation for heavy workloads.
- **Inspect Anywhere** — Every card has a built-in output viewer. Click any executed card to see its results — data tables, model summaries, metrics, charts

## Project Structure

Frontend and backend are **separate, independent projects** within a monorepo. They communicate only via REST/WebSocket APIs — no shared code, no shared runtime.

```
tensorRag/
├── frontend/               # Next.js app (TypeScript)
│   ├── package.json
│   ├── src/
│   └── ...
├── backend/                # FastAPI app (Python)
│   ├── pyproject.toml
│   ├── app/
│   ├── cards/              # Modal card functions
│   └── ...
├── docs/                   # Project documentation
└── README.md
```

## Tech Stack

| Layer | Technology | Location |
|-------|------------|----------|
| Frontend | Next.js, React Flow, Zustand | `frontend/` |
| Backend | FastAPI, Python 3.11+ | `backend/` |
| Compute | Modal (serverless) | `backend/cards/` |
| Data Transport | Object storage (TBD) | managed by backend |

## Development

```bash
# Frontend (runs on :3000)
cd frontend && npm install && npm run dev

# Backend (runs on :8000)
cd backend && pip install -e . && uvicorn app.main:app --reload
```

## Documentation

- [Architecture](docs/architecture.md) — System design and component interactions
- [Design Decisions](docs/design-decisions.md) — Key decisions and trade-offs
- [V1 Scope & Roadmap](docs/v1-roadmap.md) — MVP scope and phased plan

## Status

**Phase: Planning** — Architecture and design decisions under discussion.
