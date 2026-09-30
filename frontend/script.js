// AI Email Priority Classifier — frontend logic.
//
// Calls the FastAPI backend's /predict endpoint and renders the result.
// Same-origin relative paths are used throughout, so this works unchanged
// whether the frontend and API are served from the same Vercel project
// (see vercel.json) or from localhost during `vercel dev`.

const API_BASE = ""; // same-origin; set to e.g. "http://localhost:8000" if the API runs separately

const EXAMPLES = {
  interview:
    "Dear Applicant, your interview has been scheduled for tomorrow at 10:00 AM. Please confirm your attendance.",
  report:
    "Hi team, please review the attached quarterly report and send feedback by next week. Let me know if anything is unclear.",
  newsletter:
    "Check out our weekly newsletter for the latest company announcements, deals and promotions!",
};

const els = {
  textarea: document.getElementById("email-input"),
  charCount: document.getElementById("char-count"),
  analyzeBtn: document.getElementById("analyze-btn"),
  errorMsg: document.getElementById("error-msg"),
  resultEmpty: document.getElementById("result-empty"),
  resultBody: document.getElementById("result-body"),
  priorityFlag: document.getElementById("priority-flag"),
  flagLabel: document.getElementById("flag-label"),
  flagConfidence: document.getElementById("flag-confidence"),
  rawToggle: document.getElementById("raw-toggle"),
  rawJson: document.getElementById("raw-json"),
  chips: document.querySelectorAll(".chip"),
};

function updateCharCount() {
  const n = els.textarea.value.length;
  els.charCount.textContent = `${n} character${n === 1 ? "" : "s"}`;
}

function setLoading(isLoading) {
  els.analyzeBtn.disabled = isLoading;
  els.analyzeBtn.classList.toggle("is-loading", isLoading);
}

function showError(message) {
  els.errorMsg.textContent = message;
  els.errorMsg.hidden = false;
}

function clearError() {
  els.errorMsg.hidden = true;
  els.errorMsg.textContent = "";
}

function renderResult(data) {
  const cls = data.priority.toLowerCase(); // "high" | "medium" | "low"

  els.resultEmpty.hidden = true;
  els.resultBody.hidden = false;

  els.priorityFlag.dataset.cls = cls;
  els.flagLabel.textContent = data.priority;
  els.flagConfidence.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;

  for (const key of ["high", "medium", "low"]) {
    const p = data.probabilities[key] ?? 0;
    const fill = document.getElementById(`fill-${key}`);
    const value = document.getElementById(`value-${key}`);
    // Reset to 0 first so the width transition replays on every run.
    fill.style.width = "0%";
    requestAnimationFrame(() => {
      fill.style.width = `${(p * 100).toFixed(1)}%`;
    });
    value.textContent = p.toFixed(3);
  }

  els.rawJson.textContent = JSON.stringify(data, null, 2);
  els.rawToggle.hidden = false;
  els.rawToggle.textContent = "View raw response";
  els.rawJson.hidden = true;
}

async function analyze() {
  const email = els.textarea.value.trim();
  clearError();

  if (!email) {
    showError("Paste some email text first.");
    return;
  }

  setLoading(true);
  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email }),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      const detail =
        typeof body.detail === "string"
          ? body.detail
          : "The API rejected this request.";
      throw new Error(detail);
    }

    const data = await res.json();
    renderResult(data);
  } catch (err) {
    showError(
      err.message === "Failed to fetch"
        ? "Could not reach the API. Is it running and is API_BASE set correctly?"
        : err.message
    );
  } finally {
    setLoading(false);
  }
}

els.textarea.addEventListener("input", updateCharCount);

els.analyzeBtn.addEventListener("click", analyze);

els.textarea.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyze();
});

els.chips.forEach((chip) => {
  chip.addEventListener("click", () => {
    els.textarea.value = EXAMPLES[chip.dataset.example] || "";
    updateCharCount();
    analyze();
  });
});

els.rawToggle.addEventListener("click", () => {
  const isHidden = els.rawJson.hidden;
  els.rawJson.hidden = !isHidden;
  els.rawToggle.textContent = isHidden ? "Hide raw response" : "View raw response";
});

updateCharCount();
