"""
AI Email Priority Classifier — FastAPI backend.

Loads the TF-IDF + classifier pipeline trained in
`notebooks/email_priority_training.ipynb` and exposes:

    GET  /            basic info
    GET  /health       liveness check
    POST /predict       {"email": "..."} -> priority + confidence + probabilities

Run locally:
    uvicorn api.index:app --reload
Then open:
    http://localhost:8000/docs

Deploy on Vercel: point vercel.json at this file; Vercel's Python runtime
serves the FastAPI `app` object directly (no Mangum/adapter needed).
"""

import os
import re
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Text cleaning — MUST match notebooks/email_priority_training.ipynb Section 2
# exactly, or predictions will drift from what the model was trained on.
# ---------------------------------------------------------------------------
HEADER_LINE = re.compile(
    r"^(from|to|cc|bcc|subject|date|received|message-id|content-type|"
    r"mime-version|x-[\w-]+|return-path|reply-to):.*$",
    re.IGNORECASE | re.MULTILINE,
)
HTML_TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"https?://\S+|www\.\S+")
NON_ALPHANUM = re.compile(r"[^a-z0-9\s]")
MULTI_SPACE = re.compile(r"\s+")


def clean_email(text: str) -> str:
    """Lower-case, strip headers/HTML/URLs, collapse whitespace."""
    t = str(text)
    t = HEADER_LINE.sub(" ", t)
    t = HTML_TAG.sub(" ", t)
    t = URL.sub(" ", t)
    t = t.lower()
    t = NON_ALPHANUM.sub(" ", t)
    t = MULTI_SPACE.sub(" ", t).strip()
    return t


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "email_priority_model.pkl")

_pipeline = None
_load_error = None
try:
    _pipeline = joblib.load(MODEL_PATH)
except Exception as exc:  # noqa: BLE001 - surface any load failure via /health
    _load_error = str(exc)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Email Priority Classifier",
    description="Predicts whether an email is HIGH / MEDIUM / LOW priority.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend's origin before going live
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmailRequest(BaseModel):
    email: str = Field(..., min_length=1, description="Raw email text to classify")


class PredictResponse(BaseModel):
    priority: str
    confidence: float
    probabilities: dict


@app.get("/")
def root():
    return {
        "name": "AI Email Priority Classifier",
        "status": "ok" if _pipeline is not None else "model not loaded",
        "endpoints": ["/health", "/predict"],
    }


@app.get("/health")
def health():
    if _pipeline is None:
        return {"status": "error", "detail": _load_error}
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: EmailRequest):
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {_load_error}. "
            f"Run the training notebook first to create {MODEL_PATH}.",
        )

    cleaned = clean_email(req.email)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Email text is empty after cleaning.")

    proba = _pipeline.predict_proba([cleaned])[0]
    classes = list(_pipeline.classes_)
    probabilities = {cls: float(p) for cls, p in zip(classes, proba)}

    best_idx = int(proba.argmax())
    priority = classes[best_idx]
    confidence = float(proba[best_idx])

    return PredictResponse(
        priority=priority.upper(),
        confidence=confidence,
        probabilities=probabilities,
    )
