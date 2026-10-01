"""
AI Email Priority Classifier — FastAPI backend.

Loads the TF-IDF + classifier pipeline trained in
`notebooks/email_priority_training.ipynb` and exposes:

    GET  /            basic info
    GET  /health      liveness check
    POST /predict     {"email": "..."} -> priority + confidence + probabilities
"""

import os
import re
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Text cleaning — MUST match notebooks/email_priority_training.ipynb
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
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "model", "email_priority_model.pkl"
)

_pipeline = None
_load_error = None

try:
    if os.path.exists(MODEL_PATH):
        _pipeline = joblib.load(MODEL_PATH)
    else:
        _load_error = f"Model file not found at path: {MODEL_PATH}"
except Exception as exc:
    _load_error = str(exc)


# ---------------------------------------------------------------------------
# API App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Email Priority Classifier",
    description="Predicts whether an email is HIGH / MEDIUM / LOW priority.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class EmailRequest(BaseModel):
    email: str = Field(..., min_length=1, description="Raw email text to classify")


class PredictResponse(BaseModel):
    priority: str
    confidence: float
    probabilities: dict[str, float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "name": "AI Email Priority Classifier API",
        "status": "ok" if _pipeline is not None else "model_not_loaded",
        "error": _load_error,
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
            f"Ensure {MODEL_PATH} is committed to GitHub.",
        )

    # 1. Clean input text
    cleaned = clean_email(req.email)
    
    # Fallback to original text if regex cleaning stripped everything
    input_text = cleaned if len(cleaned) > 0 else req.email.strip()

    if not input_text:
        raise HTTPException(
            status_code=400, detail="Email text cannot be empty."
        )

    try:
        # 2. Model inference
        proba = _pipeline.predict_proba([input_text])[0]
        classes = list(_pipeline.classes_)

        # 3. Format probabilities with normalized UPPERCASE keys
        probabilities = {
            str(cls).upper(): round(float(p), 4) for cls, p in zip(classes, proba)
        }

        best_idx = int(proba.argmax())
        priority = str(classes[best_idx]).upper()
        confidence = round(float(proba[best_idx]), 4)

        return PredictResponse(
            priority=priority,
            confidence=confidence,
            probabilities=probabilities,
        )
    except Exception as err:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(err)}"
        )
