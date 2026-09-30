# AI Email Priority Classifier

Paste an email's text in, get back its priority — **high**, **medium** or
**low** — with a confidence score. TF-IDF + logistic regression, served with
FastAPI, deployed on Vercel. No LLM, no Gmail integration, no database — v1
is deliberately small.

```
Email text
    │
    ▼
FastAPI  (api/index.py)
    │
    ▼
Text cleaning (strip headers / HTML / URLs)
    │
    ▼
TF-IDF
    │
    ▼
Logistic Regression
    │
    ▼
priority + confidence + probabilities  (JSON)
    │
    ▼
Frontend  (frontend/)
```

## Project structure

```
ai-email-priority-classifier/
│
├── api/
│   ├── index.py                    FastAPI app (GET /, /health, POST /predict)
│   └── model/
│       ├── email_priority_model.pkl   trained sklearn Pipeline (created by the notebook)
│       └── model_info.json            model metadata (created by the notebook)
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── notebooks/
│   └── email_priority_training.ipynb   loads data, trains, evaluates, saves the model
│
├── data/
│   └── README.md                   where to get the dataset (not committed)
│
├── requirements.txt                 API runtime dependencies
├── .python-version
├── .gitignore
├── vercel.json
└── README.md
```

## 1. Train the model

The dataset only has spam/ham labels — the notebook derives priority labels
itself. See `data/README.md` for the full explanation and its limitations.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pandas matplotlib seaborn jupyter nbformat ipykernel

# 1. Download the dataset per data/README.md, save as data/spam_assassin.csv
# 2. Run every cell in notebooks/email_priority_training.ipynb
```

That produces `api/model/email_priority_model.pkl`, which `api/index.py`
loads at request time.

## 2. Run the API locally

```bash
uvicorn api.index:app --reload
```

- App info: http://localhost:8000/
- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"email": "Your interview is scheduled tomorrow at 10 AM."}'
```

```json
{
  "priority": "HIGH",
  "confidence": 0.947,
  "probabilities": { "high": 0.947, "medium": 0.041, "low": 0.012 }
}
```

## 3. Run the frontend locally

The frontend calls `/predict` on the same origin it's served from (see
`API_BASE` at the top of `frontend/script.js`). Simplest local setup: run the
API on port 8000, then either

- serve `frontend/` with any static server and point `API_BASE` at
  `http://localhost:8000`, or
- use `vercel dev` (below), which serves both from one origin exactly like
  production.

## 4. Deploy to Vercel

```bash
npm i -g vercel
vercel link      # first time only
vercel dev        # local dev matching production routing
vercel --prod     # deploy
```

`vercel.json` routes `/predict`, `/health` and the FastAPI docs routes to
`api/index.py`, and everything else to the static files in `frontend/`. Make
sure `api/model/email_priority_model.pkl` is committed — Vercel does not run
the training notebook at build time (see the note in `.gitignore`).

## Limitations

- **Priority labels are rule-based, not ground truth.** See `data/README.md`.
  Treat this as a v1 portfolio project, not a validated priority model.
- **No personalization.** The model has no notion of who "you" are — it
  scores generic urgency signals in the text.
- **CORS is wide open (`allow_origins=["*"]`)** in `api/index.py` for easy
  local testing. Tighten it to your actual frontend origin before sharing
  the API publicly.

## Roadmap (not built yet)

- v2: a second head predicting a category (career, finance, work, personal).
- v3: an LLM layer that explains *why* an email got its priority.

See the original project brief for the full reasoning behind these staged cuts.
