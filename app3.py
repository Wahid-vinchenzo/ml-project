import streamlit as st
import torch
import pandas as pd
import numpy as np
from pathlib import Path
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except (ImportError, AttributeError):
    try:
        from transformers.models.auto import AutoTokenizer, AutoModelForSequenceClassification
    except (ImportError, AttributeError):
        try:
            from transformers.models.auto.tokenization_auto import AutoTokenizer
            from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
        except Exception:
            AutoTokenizer = None
            AutoModelForSequenceClassification = None
from sentence_transformers import SentenceTransformer
from torch.nn.functional import softmax
import joblib
import json
from docx import Document
import io
import re

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="ToS Analyzer",
    page_icon="📄",
    layout="wide"
)

# =========================
# Device setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Paths & settings
# =========================
_current_dir = Path(__file__).resolve().parent
if (_current_dir / "models").exists():
    MODELS_DIR = _current_dir / "models"
elif (_current_dir / "tos-analyzer-streamlit-main" / "models").exists():
    MODELS_DIR = _current_dir / "tos-analyzer-streamlit-main" / "models"
else:
    MODELS_DIR = _current_dir / "models"
THRESHOLD = 0.40  # Fixed threshold

def is_git_lfs_pointer(file_path: Path) -> bool:
    """Check if a file is a Git LFS pointer text file instead of real binary weights."""
    try:
        if file_path.exists() and file_path.stat().st_size < 1024:
            with open(file_path, "r", errors="ignore") as f:
                content = f.read(150)
                if "git-lfs" in content:
                    return True
    except Exception:
        pass
    return False

# =========================
# Low-info filtering helpers
# =========================
STOPWORDS = set()

GENERIC_SHORT = set([
    "privacy policy",
    "platform",
    "support for users",
    "press & media center"
])

def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

def is_low_info_hit(text: str, min_words: int = 10, min_chars: int = 40) -> bool:
    """
    Returns True if text is likely a heading/fragment that is not informative,
    even if the model is confident.
    """
    t = normalize_text(text)
    if not t:
        return True

    tl = t.lower()

    # 1) Generic heading exact match
    if tl in GENERIC_SHORT:
        return True

    # 2) Too short in chars
    if len(t) < min_chars:
        return True

    # 3) Too few word tokens (letters only)
    words = re.findall(r"[A-Za-z]+", t)
    if len(words) < min_words:
        return True

    # 4) Mostly stopwords -> not informative
    content_words = [w.lower() for w in words if w.lower() not in STOPWORDS]
    if len(content_words) <= 2:
        return True

    # 5) ALL CAPS short heading
    letters = re.findall(r"[A-Za-z]", t)
    if letters:
        upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
        if upper_ratio > 0.85 and len(words) <= 12:
            return True

    return False

# =========================
# Cache model loading
# =========================
@st.cache_resource
def load_models():
    """Load all three models and ensemble metadata"""
    meta_path = MODELS_DIR / "ensemble_meta_v2.json"
    if not meta_path.exists():
        st.error(f"Metadata file not found at {meta_path}")
        return None, [], torch.tensor([])

    # Load ensemble metadata
    with open(meta_path, "r") as f:
        meta = json.load(f)

    classes = meta["classes"]
    weights = torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32)

    # Check if weights are Git LFS pointer files
    lfs_files = []
    for check_file in [
        MODELS_DIR / "LegalBERT_full_v2" / "model.safetensors",
        MODELS_DIR / "DistilBERT_full_v2" / "model.safetensors",
        MODELS_DIR / "SBERT_full_v2" / "sbert_encoder" / "model.safetensors",
        MODELS_DIR / "SBERT_full_v2" / "sbert_lr_v2.joblib",
    ]:
        if is_git_lfs_pointer(check_file):
            lfs_files.append(check_file.name)

    if lfs_files:
        return "LFS_POINTERS_DETECTED", classes, weights

    if AutoTokenizer is None or AutoModelForSequenceClassification is None:
        return "TRANSFORMERS_UNAVAILABLE", classes, weights

    try:
        models = {}

        # Load LegalBERT
        legal_path = MODELS_DIR / "LegalBERT_full_v2"
        models["legal_tok"] = AutoTokenizer.from_pretrained(str(legal_path), local_files_only=True)
        models["legal_model"] = AutoModelForSequenceClassification.from_pretrained(
            str(legal_path), local_files_only=True
        ).to(device).eval()

        # Load DistilBERT
        distil_path = MODELS_DIR / "DistilBERT_full_v2"
        models["distil_tok"] = AutoTokenizer.from_pretrained(str(distil_path), local_files_only=True)
        models["distil_model"] = AutoModelForSequenceClassification.from_pretrained(
            str(distil_path), local_files_only=True
        ).to(device).eval()

        # Load SBERT
        sbert_path = MODELS_DIR / "SBERT_full_v2"
        models["sbert"] = SentenceTransformer(str(sbert_path / "sbert_encoder"), device=str(device))
        models["sbert_clf"] = joblib.load(str(sbert_path / "sbert_lr_v2.joblib"))

        return models, classes, weights
    except Exception as e:
        return f"ERROR: {str(e)}", classes, weights

# =========================
# Document parsing
# =========================
def extract_paragraphs_from_docx(file):
    """Extract paragraphs from uploaded .docx file"""
    doc = Document(io.BytesIO(file.read()))
    paragraphs = []
    for p in doc.paragraphs:
        text = normalize_text(p.text)
        if text:
            paragraphs.append(text)
    return paragraphs

# =========================
# Model inference helpers
# =========================
def predict_hf(texts, tok, model, batch_size=32, max_length=256):
    """Predict using HuggingFace models (LegalBERT, DistilBERT)"""
    all_logits = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
            all_logits.append(logits.cpu())

    logits = torch.cat(all_logits, dim=0)
    probs = softmax(logits, dim=-1).numpy()
    return probs

def predict_sbert(texts, sbert, clf, batch_size=256):
    """Predict using SBERT + Logistic Regression"""
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = sbert.encode(
            batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False
        )
        all_vecs.append(emb)

    X = np.vstack(all_vecs)
    probs = clf.predict_proba(X)
    return probs

def get_heuristic_predictions(texts, classes):
    """Heuristic keyword-based classifier for demo mode when weights are LFS pointers"""
    keywords = {
        "arbitration": ["arbitrat", "dispute", "class action waiver", "jury trial", "binding arbitration", "tribunal", "adr"],
        "content_license": ["royalty-free", "perpetual", "worldwide license", "irrevocable", "sublicensable", "grant the company", "ownership of content", "moral rights"],
        "limitation_liability": ["limitation of liability", "liable", "damages", "consequential", "punitive damages", "as is", "warranty disclaimer", "hold harmless", "maximum extent permitted"],
        "privacy": ["personal data", "personally identifiable", "privacy policy", "cookies", "third-party tracking", "data sharing", "sell your data", "beacons"],
        "prohibited_use": ["prohibited", "unlawful", "reverse engineer", "hack", "circumvent", "restricted activities", "violate laws", "infringe"],
        "termination": ["terminate", "suspend", "without notice", "discontinue", "sole discretion", "cancel your account", "immediate effect"]
    }

    n_samples = len(texts)
    n_classes = len(classes)
    probs = np.full((n_samples, n_classes), 0.05, dtype=np.float32)

    for i, t in enumerate(texts):
        tl = t.lower()
        matched = False
        for c_idx, cls in enumerate(classes):
            words = keywords.get(cls, [])
            match_count = sum(1 for w in words if w in tl)
            if match_count > 0:
                probs[i, c_idx] = min(0.95, 0.45 + match_count * 0.20)
                matched = True
        row_sum = probs[i].sum()
        if row_sum > 0:
            probs[i] /= row_sum
        if not matched:
            probs[i] = np.full(n_classes, 1.0 / n_classes)

    return probs

def get_ensemble_predictions(texts, models, classes, weights):
    """Get ensemble predictions from all three models"""
    if models == "LFS_POINTERS_DETECTED" or isinstance(models, str) or models is None:
        return get_heuristic_predictions(texts, classes)

    legal_probs = predict_hf(texts, models["legal_tok"], models["legal_model"])
    distil_probs = predict_hf(texts, models["distil_tok"], models["distil_model"])
    sbert_probs = predict_sbert(texts, models["sbert"], models["sbert_clf"])

    # (3, N, C)
    member_probs = np.stack([legal_probs, distil_probs, sbert_probs], axis=0)

    # Weighted ensemble => (N, C)
    ens_probs = np.tensordot(weights.numpy(), member_probs, axes=([0], [0]))
    return ens_probs

def format_violation_type(label):
    """Format violation label for display"""
    label_map = {
        "arbitration": "⚖️ Arbitration Clause",
        "content_license": "📝 Content License Issue",
        "limitation_liability": "🛡️ Limitation of Liability",
        "privacy": "🔒 Privacy Concern",
        "prohibited_use": "🚫 Prohibited Use",
        "termination": "❌ Termination Clause"
    }
    return label_map.get(label, label)

def get_conf_level_class(confidence: float) -> str:
    """Classify confidence score for badge color styling"""
    if confidence >= 0.70:
        return "conf-high"
    elif confidence >= 0.50:
        return "conf-med"
    return "conf-low"

CUSTOM_CSS = """
<style>
/* ============================================================
   LEGALTECH SAAS THEME - REFINED DARK PALETTE
   ============================================================ */

/* App container layout */
.main .block-container {
    max-width: 1100px !important;
    padding-top: 1.8rem !important;
    padding-bottom: 3.5rem !important;
}

/* Header styling */
.main-header {
    background: linear-gradient(180deg, #151d30 0%, #0d1322 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 2.2rem 2.2rem 1.9rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 25px -8px rgba(0, 0, 0, 0.35);
}
.header-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.25);
    padding: 0.22rem 0.7rem;
    border-radius: 20px;
    margin-bottom: 0.85rem;
    text-transform: uppercase;
}
.header-title {
    font-size: 2.15rem !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
    margin: 0 0 0.5rem 0 !important;
    letter-spacing: -0.025em !important;
    line-height: 1.25 !important;
    display: flex;
    align-items: center;
    gap: 0.65rem;
}
.header-icon {
    font-size: 1.95rem;
    line-height: 1;
}
.header-subtitle {
    font-size: 1.02rem !important;
    color: #94a3b8 !important;
    margin: 0 !important;
    line-height: 1.55 !important;
    max-width: 760px;
}

/* Info Guide Grid inside Expander */
.info-guide-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 0.85rem;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}
.info-guide-item {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 0.85rem 1rem;
}
.info-tag {
    font-size: 0.88rem;
    font-weight: 600;
    color: #f1f5f9;
    display: block;
    margin-bottom: 0.35rem;
}
.info-guide-item p {
    font-size: 0.82rem;
    color: #94a3b8;
    margin: 0;
    line-height: 1.45;
}

/* Document status pill */
.doc-status-card {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: #161e2e;
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 10px;
    padding: 0.85rem 1.25rem;
    margin: 1.25rem 0 0.5rem;
    color: #e2e8f0;
}
.doc-status-icon {
    font-size: 1.3rem;
}
.doc-status-text {
    font-size: 0.95rem;
}
.doc-status-text strong {
    color: #f8fafc;
}
.doc-status-text span {
    color: #38bdf8;
    font-weight: 500;
}

/* Noise Control Container */
.noise-card {
    background: #151c2c;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.25rem 1.5rem 0.5rem;
    margin: 1.5rem 0 1.25rem;
}
.noise-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 0.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.noise-desc {
    font-size: 0.83rem;
    color: #94a3b8;
    margin-bottom: 0.75rem;
}
.control-box-checkbox {
    padding-top: 1.85rem;
}

/* Analysis Results Section Header */
.section-header-block {
    margin-top: 2rem;
    margin-bottom: 1.25rem;
}
.section-heading {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
    margin: 0 0 0.35rem 0 !important;
    letter-spacing: -0.015em !important;
}
.section-subheading {
    font-size: 0.92rem;
    color: #94a3b8;
}

/* Streamlit Metric Cards */
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, #161e2e 0%, #111827 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 1.25rem 1.4rem !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
    transition: all 0.2s ease-in-out !important;
}
div[data-testid="stMetric"]:hover {
    border-color: rgba(56, 189, 248, 0.3) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35) !important;
}
div[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}
div[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-size: 2.15rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

/* Alert bar for issues found */
.issues-alert-bar {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-radius: 10px;
    padding: 0.85rem 1.25rem;
    margin: 1.5rem 0 1rem;
    color: #fca5a5;
    font-size: 0.95rem;
}
.issues-alert-bar strong {
    color: #fecaca;
}

/* Problematic Clause Card */
.clause-card {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.85rem;
    transition: border-color 0.2s ease;
}
.clause-card:hover {
    border-color: rgba(148, 163, 184, 0.25);
}
.clause-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
}
.para-pill {
    background: rgba(148, 163, 184, 0.12);
    color: #cbd5e1;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.22rem 0.65rem;
    border-radius: 6px;
    letter-spacing: 0.02em;
    border: 1px solid rgba(148, 163, 184, 0.2);
}
.conf-badge {
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.22rem 0.65rem;
    border-radius: 6px;
}
.conf-high {
    background: rgba(239, 68, 68, 0.14);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}
.conf-med {
    background: rgba(245, 158, 11, 0.14);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}
.conf-low {
    background: rgba(56, 189, 248, 0.14);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
}
.clause-text {
    background: #0b0f19;
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1.15rem;
    color: #e2e8f0;
    font-size: 0.925rem;
    line-height: 1.6;
}

/* Streamlit Expander styling */
div[data-testid="stExpander"] {
    background: #161e2e !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    margin-bottom: 0.85rem !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15) !important;
    overflow: hidden !important;
}
div[data-testid="stExpander"]:hover {
    border-color: rgba(56, 189, 248, 0.25) !important;
}
div[data-testid="stExpander"] summary {
    font-size: 1.02rem !important;
    font-weight: 600 !important;
    color: #f1f5f9 !important;
    padding: 0.85rem 1.25rem !important;
}

/* File Uploader styling */
div[data-testid="stFileUploader"] {
    background: #151c2c;
    border: 1.5px dashed rgba(148, 163, 184, 0.25);
    border-radius: 14px;
    padding: 1.25rem 1.5rem 1rem;
    transition: all 0.2s ease;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(56, 189, 248, 0.5);
    background: #172033;
}

/* Download button */
.download-container {
    margin-top: 1.25rem;
    margin-bottom: 1.5rem;
}
div[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    border-radius: 10px !important;
    padding: 0.7rem 1.65rem !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    box-shadow: 0 4px 15px rgba(2, 132, 199, 0.25) !important;
    transition: all 0.2s ease-in-out !important;
}
div[data-testid="stDownloadButton"] button:hover {
    background: linear-gradient(135deg, #0369a1 0%, #1d4ed8 100%) !important;
    box-shadow: 0 6px 22px rgba(2, 132, 199, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* Clean document celebration card */
.safe-celebration-card {
    background: linear-gradient(180deg, rgba(16, 185, 129, 0.08) 0%, rgba(16, 185, 129, 0.02) 100%);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 14px;
    padding: 2.25rem 2rem;
    text-align: center;
    margin: 1.75rem 0;
}
.safe-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
}
.safe-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #34d399;
    margin-bottom: 0.5rem;
}
.safe-desc {
    color: #94a3b8;
    font-size: 0.98rem;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.55;
}

/* Responsive Overrides */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .main-header {
        padding: 1.5rem 1.25rem !important;
    }
    .header-title {
        font-size: 1.6rem !important;
    }
    .header-subtitle {
        font-size: 0.92rem !important;
    }
    .control-box-checkbox {
        padding-top: 0.25rem !important;
        margin-bottom: 0.5rem;
    }
    .clause-header {
        flex-direction: column !important;
        align-items: flex-start !important;
        gap: 0.45rem !important;
    }
    div[data-testid="stMetric"] {
        margin-bottom: 0.75rem !important;
    }
}
</style>
"""

# =========================
# Main app
# =========================
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header">
        <div class="header-badge">Legal Contract Auditing Engine</div>
        <h1 class="header-title"><span class="header-icon">📄</span> Terms of Service Analyzer</h1>
        <p class="header-subtitle">Upload your Terms of Service or Terms & Conditions document to check for potentially problematic clauses.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    with st.spinner("Loading AI models... (this may take a minute)"):
        models, classes, weights = load_models()

    if models == "LFS_POINTERS_DETECTED":
        st.warning("⚠️ **Git LFS Pointer Stubs Detected in `models/`**")
        st.info(
            "The model weights (`model.safetensors`, `sbert_lr_v2.joblib`) are small Git LFS pointer stubs (~134 bytes) "
            "rather than downloaded model weights (~800 MB total).\n\n"
            "This happens when a repository is downloaded from GitHub as a ZIP archive instead of cloned with `git lfs`.\n\n"
            "👉 **Operating in Demo Mode (Heuristic Keyword Classifier)**: You can upload documents and test all app features and UI components right now."
        )
    elif models == "TRANSFORMERS_UNAVAILABLE":
        st.warning("⚠️ **Transformers Module Initialization Notice**")
        st.info("Operating in **Demo Mode (Heuristic Keyword Classifier)**.")
    elif isinstance(models, str) and models.startswith("ERROR"):
        st.error(f"Failed to load models: {models}")
        st.info("Operating in Demo Mode.")
    else:
        st.success("✅ Ready to analyze documents with AI ensemble models!")

    # Info box
    with st.expander("ℹ️ What does this tool do?"):
        st.markdown("""
        <div class="info-guide-grid">
            <div class="info-guide-item">
                <span class="info-tag">⚖️ Arbitration</span>
                <p>Forced mandatory arbitration and class-action waivers restricting access to court proceedings.</p>
            </div>
            <div class="info-guide-item">
                <span class="info-tag">📝 Content License</span>
                <p>Broad, perpetual, or royalty-free rights granted to the company over user-generated content.</p>
            </div>
            <div class="info-guide-item">
                <span class="info-tag">🛡️ Limitation of Liability</span>
                <p>Excessive disclaimers of company responsibility, indirect damages, and warranties.</p>
            </div>
            <div class="info-guide-item">
                <span class="info-tag">🔒 Privacy</span>
                <p>Concerning user data practices, third-party distribution, and extensive behavioral tracking.</p>
            </div>
            <div class="info-guide-item">
                <span class="info-tag">🚫 Prohibited Use</span>
                <p>Ambiguous or restrictive constraints on user activity and software reverse-engineering.</p>
            </div>
            <div class="info-guide-item">
                <span class="info-tag">❌ Termination</span>
                <p>Unfair, discretionary account suspension or termination conditions without prior notice.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Upload
    uploaded_file = st.file_uploader(
        "Choose a .docx file",
        type=["docx"],
        help="Upload a Terms of Service or Terms & Conditions document in .docx format"
    )

    if uploaded_file is None:
        return

    # Extract paragraphs
    with st.spinner("Reading document..."):
        paragraphs = extract_paragraphs_from_docx(uploaded_file)

    st.markdown(f"""
    <div class="doc-status-card">
        <span class="doc-status-icon">📑</span>
        <div class="doc-status-text">
            <strong>{uploaded_file.name}</strong> loaded &bull; <span>{len(paragraphs)} paragraphs detected</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if len(paragraphs) == 0:
        st.warning("No readable paragraphs found in the uploaded file.")
        return

    # Low-info filtering controls
    st.markdown("""
    <div class="noise-card">
        <div class="noise-title">🧹 Noise control (optional)</div>
        <div class="noise-desc">Fine-tune detection sensitivity to exclude short headings, structural labels, and low-information clauses.</div>
    """, unsafe_allow_html=True)

    colA, colB, colC = st.columns([1.1, 1.2, 1.2])
    with colA:
        st.markdown('<div class="control-box-checkbox">', unsafe_allow_html=True)
        show_low_info = st.checkbox("Show low-info hits", value=False)
        st.markdown('</div>', unsafe_allow_html=True)
    with colB:
        min_words = st.slider("Min words", 3, 25, 10)
    with colC:
        min_chars = st.slider("Min chars", 10, 200, 40)

    st.markdown('</div>', unsafe_allow_html=True)

    # Predictions
    with st.spinner("Analyzing document with AI models..."):
        ens_probs = get_ensemble_predictions(paragraphs, models, classes, weights)

    max_probs = ens_probs.max(axis=1)
    pred_ids = ens_probs.argmax(axis=1)

    # Filter violations + hide low-info by default
    violations = []
    low_info_hits = []

    for idx, (text, max_prob, pred_id) in enumerate(zip(paragraphs, max_probs, pred_ids)):
        if max_prob < THRESHOLD:
            continue

        violation_type = classes[pred_id]
        hit = {
            "paragraph": idx + 1,
            "type": violation_type,
            "text": text,
            "confidence": float(max_prob)
        }

        low_info = is_low_info_hit(text, min_words=min_words, min_chars=min_chars)
        if low_info and not show_low_info:
            low_info_hits.append(hit)
            continue

        violations.append(hit)

    # Display results
    st.markdown("""
    <div class="section-header-block">
        <h2 class="section-heading">📊 Analysis Results</h2>
        <div class="section-subheading">Comprehensive contract safety breakdown and detected clause categories.</div>
    </div>
    """, unsafe_allow_html=True)

    total = len(paragraphs)
    violation_count = len(violations)
    safe_count = total - violation_count

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Paragraphs", total)
    col2.metric("✅ Safe", safe_count, delta=f"{safe_count/total*100:.1f}%")
    col3.metric("⚠️ Issues Found", violation_count, delta=f"{violation_count/total*100:.1f}%", delta_color="inverse")

    if len(low_info_hits) > 0 and not show_low_info:
        st.caption(f"🧹 Hid {len(low_info_hits)} low-info hit(s). Enable **Show low-info hits** to review them.")

    # Show violations
    if violation_count > 0:
        st.markdown(f"""
        <div class="issues-alert-bar">
            <span>⚠️</span> Found <strong>{violation_count}</strong> potentially problematic clause(s) in this document:
        </div>
        """, unsafe_allow_html=True)

        # Group by type
        violation_types = {}
        for v in violations:
            v_type = v["type"]
            violation_types.setdefault(v_type, []).append(v)

        # Display per category
        for v_type, items in violation_types.items():
            with st.expander(f"{format_violation_type(v_type)} ({len(items)} found)", expanded=True):
                for item in items:
                    conf_cls = get_conf_level_class(item['confidence'])
                    st.markdown(f"""
                    <div class="clause-card">
                        <div class="clause-header">
                            <span class="para-pill">§ Paragraph {item['paragraph']}</span>
                            <span class="conf-badge {conf_cls}">Confidence: {item['confidence']:.1%}</span>
                        </div>
                        <div class="clause-text">
                            {item['text']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Summary table
        st.markdown("""
        <div class="section-header-block" style="margin-top: 1.5rem; margin-bottom: 0.75rem;">
            <h3 style="color: #f8fafc; font-size: 1.25rem; font-weight: 600; margin: 0;">📋 Summary</h3>
        </div>
        """, unsafe_allow_html=True)
        summary_df = pd.DataFrame(
            [{"Violation Type": format_violation_type(k), "Count": len(v)} for k, v in violation_types.items()]
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Download option
        csv_df = pd.DataFrame([{
            "Paragraph": v["paragraph"],
            "Violation Type": v["type"],
            "Confidence": f"{v['confidence']:.2%}",
            "Text": v["text"]
        } for v in violations])

        csv = csv_df.to_csv(index=False)
        st.markdown('<div class="download-container">', unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Detailed Report (CSV)",
            data=csv,
            file_name=f"{uploaded_file.name}_violations.csv",
            mime="text/csv"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="safe-celebration-card">
            <div class="safe-icon">🎉</div>
            <div class="safe-title">Clean Document Assessment</div>
            <p class="safe-desc">Great news! No problematic clauses detected in this document.</p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()

        # Optional: if everything got filtered as low-info but threshold triggered,
        # you can still hint user to enable view.
        if len(low_info_hits) > 0 and not show_low_info:
            st.info("Some hits were filtered as low-info. Enable **Show low-info hits** if you want to review them.")

if __name__ == "__main__":
    main()
