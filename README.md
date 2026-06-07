# Text Generation Flask Deployment

Production-oriented LSTM text generation service trained on *Harry Potter and the Sorcerer's Stone*, served via Flask and Gunicorn.

## Problem

Demonstrate the full ML lifecycle: train a language model, serialize artifacts, and serve predictions through a web API with health checks.

## Approach

| Component | Role |
|-----------|------|
| `train.py` | Train 5-gram LSTM and export weights/vocab |
| `generator.py` | Model definition and inference with lazy loading |
| `config.py` | Environment-driven hyperparameters |
| `app.py` | Flask routes, error handling, health endpoint |
| `templates/` | Minimal web UI |

Design choices:

- Lazy model load to fail fast with clear errors when artifacts are missing
- Top-k sampling for diverse but coherent generation
- Separate training and serving code paths

## Reproducibility

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make setup
python train.py          # creates model_dict and pickle files
make run                 # starts Flask on port 8080
curl localhost:8080/health
```

Production:

```bash
gunicorn app:app --bind 0.0.0.0:8080
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` | `8080` | HTTP port |
| `MODEL_PATH` | `model_dict` | Trained weights |
| `GENERATION_STEPS` | `20` | Tokens after seed |

## Tech stack

Python 3, PyTorch, Flask, Gunicorn, NLTK

## Limitations and next steps

- Store model artifacts in S3/GCS, not git
- Add load testing and request timeouts
- Replace n-gram seeding with subword tokenization for better generalization
