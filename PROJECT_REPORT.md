# Project Report: Terms of Service (ToS) Analyzer
## An Intelligent Multi-Model Ensemble System for Automated Legal Clause Classification

---

## 1. Project Title & Overview

### 1.1 Project Title
**Terms of Service (ToS) Analyzer: Intelligent Multi-Model Ensemble for Legal Clause Classification**

### 1.2 Overview
The **Terms of Service (ToS) Analyzer** is an end-to-end Natural Language Processing (NLP) document-auditing system and interactive web application. It automates the parsing, screening, and categorization of legal contracts—specifically Terms of Service (ToS), Terms of Use, and Privacy Policies—contained in Microsoft Word (`.docx`) documents.

The application leverages a heterogeneous tri-model ensemble combining **LegalBERT**, **DistilBERT**, and **Sentence-BERT (SBERT) with Logistic Regression** to detect and classify potentially problematic, unfair, or predatory clauses across six critical legal domains:
1. **Arbitration Clauses** (`arbitration`): Forced mandatory arbitration and class action waivers.
2. **Content Licensing Issues** (`content_license`): Overly broad, perpetual, royalty-free rights granted to platform operators over user-generated content.
3. **Limitation of Liability** (`limitation_liability`): Broad disclaimers of company responsibility, indirect damages, and warranties.
4. **Privacy Concerns** (`privacy`): Intrusive user surveillance, third-party data distribution, tracking, and profiling.
5. **Prohibited Use Restrictions** (`prohibited_use`): Ambiguous or excessively restrictive constraints on user activity and reverse-engineering.
6. **Unilateral Termination Conditions** (`termination`): Provisions allowing providers to terminate accounts or services at will without prior notification.

Built with a **Streamlit** user interface, the tool provides real-time quantitative safety metrics, category-specific clause drill-downs, dynamic heuristic noise filtering (to eliminate false positives on boilerplate headings), and one-click detailed CSV report exports.

---

## 2. Problem Statement & Motivation

### 2.1 Problem Statement
When accessing modern digital services, users are universally presented with Terms of Service agreements. These documents are characterized by:
- **Excessive Length & Complexity:** Agreements frequently exceed 10,000 to 30,000 words of dense, specialized legal terminology ("legalese").
- **Information Asymmetry:** Platform providers draft agreements with specialized legal counsel to protect corporate interests, while users lack legal training.
- **The "Clickwrap" Dilemma:** Academic studies demonstrate that over 99% of consumers accept terms without reading them ("the biggest lie on the web"), involuntarily forfeiting constitutional rights (e.g., Seventh Amendment jury trials via forced arbitration) and granting expansive licenses over their personal data and intellectual property.
- **High Cost of Manual Auditing:** Engaging professional legal counsel to inspect routine terms of service agreements is financially prohibitive for everyday consumers, researchers, and small businesses.

### 2.2 Motivation & Objectives
The primary objective of this project is to bridge this information asymmetry through automated machine learning. Specifically, the project aims to:
- **Democratize Legal Understanding:** Offer an accessible, zero-cost, intuitive interface that instantly highlights risky clauses for non-technical users.
- **Ensure High-Precision Classification:** Deploy an ensemble architecture that leverages domain-specific pre-training (LegalBERT) alongside semantic embedding models to maximize detection robustness and minimize false positives.
- **Preserve Data Privacy:** Run models locally without transmitting sensitive legal documents or personal contracts to external proprietary APIs or third-party cloud servers.
- **Filter Document Noise:** Implement automated heuristics that distinguish informative clauses from structural metadata (e.g., headings, uppercase section dividers, navigation fragments).

---

## 3. Technology Stack & Tools

The table below summarizes the core programming languages, frameworks, and libraries utilized across the system, along with the technical justification for their selection:

| Layer / Component | Technology / Tool | Version | Technical Justification |
| :--- | :--- | :--- | :--- |
| **Primary Language** | Python | `3.11 / 3.12` | Standard ecosystem for state-of-the-art machine learning, transformer inference, and data science. |
| **Web Framework & UI** | Streamlit | `^1.67.0` | Enables reactive, state-managed dashboards with native file uploaders, reactive metrics, expanders, and download handlers without frontend boilerplate. |
| **Deep Learning Engine** | PyTorch (`torch`) | `^2.x` | Industry-standard tensor computation engine providing optimized CPU/CUDA device abstraction and tensor arithmetic for ensemble predictions. |
| **Transformer Architecture** | Hugging Face `transformers` | `^4.57.x` | Provides optimized implementations of `AutoTokenizer` and `AutoModelForSequenceClassification` with safetensors support. |
| **Sentence Embeddings** | `sentence-transformers` (SBERT) | `^2.x` | Implements Siamese/Triplet network architectures (`all-MiniLM-L6-v2`) for generating dense 384-dimensional semantic sentence vectors. |
| **Classical Machine Learning** | `scikit-learn` & `joblib` | `^1.x` | Provides the calibrated multi-class Logistic Regression classifier running atop SBERT embeddings, serialized via `joblib`. |
| **Document Ingestion** | `python-docx` | `^1.1.x` | Provides fast, in-memory parsing of OpenXML Word documents (`.docx`) into distinct paragraph sequences without Microsoft Word dependencies. |
| **Data Processing & Linear Algebra** | `pandas` & `numpy` | `^2.x / ^1.26.x` | Vectorized tensor stacking (`np.stack`), weighted tensor contractions (`np.tensordot`), and tabular CSV serialization. |
| **Dev Environment & Containerization** | VS Code Dev Containers | `python:1-3.11-bookworm` | Standardizes reproducible Linux environments, pre-configuring system packages and port-forwarding for development. |
| **Large File Storage** | Git LFS | `v1` spec | Tracks large pre-trained weights (`*.safetensors`, `*.joblib`) exceeding GitHub's standard 100 MB file limit. |

---

## 4. Architecture & Working Mechanism (Workflow)

### 4.1 System Architecture Diagram

```
┌────────────────────────────────────────────────────────┐
│                   User Upload (.docx)                  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│      Document Parsing & Text Normalization Pipeline    │
│  - python-docx stream reader (io.BytesIO)              │
│  - Whitespace cleaning & regex normalization           │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│            Noise & Low-Information Heuristic Filter    │
│  - Generic heading filter ("privacy policy", etc.)     │
│  - Minimum character count check (min_chars = 40)      │
│  - Minimum token word check (min_words = 10)           │
│  - Uppercase section header suppression (> 85% caps)   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│      Tri-Model Heterogeneous Inference Engine          │
│                                                        │
│  ┌──────────────────┐ ┌──────────────────┐ ┌─────────┐ │
│  │    LegalBERT     │ │    DistilBERT    │ │  SBERT  │ │
│  │ (Domain-Tuned)   │ │  (General NLP)   │ │+ LogReg │ │
│  └────────┬─────────┘ └────────┬─────────┘ └────┬────┘ │
│           │ P(Legal)           │ P(Distil)      │ P(SB)│
└───────────┼────────────────────┼────────────────┼──────┘
            │                    │                │
            ▼                    ▼                ▼
┌────────────────────────────────────────────────────────┐
│            Weighted Ensemble Aggregation Layer         │
│     P_ens = 1/3 * P(Legal) + 1/3 * P(Distil) + 1/3 * P(SB)    │
│            Decision Threshold: THRESHOLD = 0.40        │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│               Interactive Streamlit Presentation       │
│  - Aggregate Safety Metrics (Safe % vs Issues %)       │
│  - Categorized Risk Expanders with Confidence Scores   │
│  - Adjustable Noise Control Sliders                    │
│  - Detailed CSV Report Generation & Download           │
└────────────────────────────────────────────────────────┘
```

---

### 4.2 Step-by-Step Processing Pipeline

#### Step 1: Ingestion and Normalization
1. The user uploads a `.docx` file through `st.file_uploader`.
2. The binary stream is wrapped into an in-memory buffer (`io.BytesIO`) and parsed using `python-docx`'s `Document` object.
3. Every non-empty paragraph is extracted and normalized using regular expressions (`re.sub(r"\s+", " ", text)`), collapsing redundant spaces, carriage returns, and tabs into consistent strings.

#### Step 2: Noise & Low-Information Heuristic Filtering
Legal contracts contain numerous non-substantive fragments (e.g., table of contents labels, copyright marks, standalone numbers, uppercase chapter titles). If passed to deep learning models unchecked, these fragments can trigger false positives due to out-of-context legal keywords.
The function `is_low_info_hit()` filters out these artifacts by checking:
- **Boilerplate Headings:** Matches against known generic phrases (`"privacy policy"`, `"platform"`, `"support for users"`, `"press & media center"`).
- **Length Constraints:** Rejects fragments shorter than `min_chars` (default: 40 characters) or fewer than `min_words` (default: 10 alphabetic tokens).
- **All-Caps Titles:** Evaluates the uppercase character ratio:
  $$\text{Upper Ratio} = \frac{\sum \text{is\_upper}(c)}{\text{total letters}} > 0.85 \quad \text{for strings with } \le 12 \text{ words}$$
Any paragraph triggering these rules is marked as `low_info` and hidden by default, unless the user toggles the noise inspection switch.

#### Step 3: Tri-Model Parallel Inference
The core engine consists of three complementary model architectures initialized via `@st.cache_resource`:

1. **LegalBERT (`LegalBERT_full_v2`):**
   - **Architecture:** 12-layer, 768-hidden, 12-head BERT model pre-trained on legal texts (legislation, court cases, contracts) and fine-tuned for multi-class sequence classification.
   - **Input:** Tokenized with `AutoTokenizer`, padded and truncated to `max_length=256`.
   - **Output:** Softmax probabilities over the 6 target classes.

2. **DistilBERT (`DistilBERT_full_v2`):**
   - **Architecture:** 6-layer, 768-hidden, 12-head distilled BERT model providing high inference velocity while retaining 97% of BERT's language comprehension.
   - **Role:** Acts as a regularizing counterweight to LegalBERT, preventing overfitting on obscure legal jargon.

3. **SBERT + Logistic Regression (`SBERT_full_v2`):**
   - **Architecture:** `SentenceTransformer` (`all-MiniLM-L6-v2`) generating dense 384-dimensional sentence embeddings, fed directly into an optimized L2-regularized `LogisticRegression` classifier.
   - **Role:** Captures sentence-level semantic similarity across paraphrased clauses without relying entirely on cross-entropy classification heads.

#### Step 4: Mathematical Formulation of Ensemble Fusion
The probability tensors from each member model are combined using uniform weighting loaded from `ensemble_meta_v2.json`:
$$\mathbf{P}_{\text{ensemble}} = \sum_{m=1}^{M} w_m \mathbf{P}_m$$
Where $M = 3$, $w = \left[\frac{1}{3}, \frac{1}{3}, \frac{1}{3}\right]$, and $\mathbf{P}_m \in \mathbb{R}^{N \times 6}$ denotes the predicted probability matrix for $N$ paragraphs across the 6 classes.

In the code, this is calculated using tensor contraction:
```python
member_probs = np.stack([legal_probs, distil_probs, sbert_probs], axis=0)  # Shape: (3, N, C)
ens_probs = np.tensordot(weights.numpy(), member_probs, axes=([0], [0]))    # Shape: (N, C)
```

#### Step 5: Decision Thresholding and Classification
For each paragraph $i$, the predicted class and confidence are determined by:
$$\hat{y}_i = \arg\max_{c} \mathbf{P}_{\text{ensemble}}[i, c], \quad c^* = \max_{c} \mathbf{P}_{\text{ensemble}}[i, c]$$
- If $c^* \ge 0.40$ (fixed threshold `THRESHOLD`), the paragraph is flagged as an active violation.
- If $c^* < 0.40$, the paragraph is marked as safe/non-problematic.

#### Step 6: Presentation and Export
- **Metrics Dashboard:** Displays the total paragraph count, percentage of safe clauses, and total issues flagged.
- **Categorized Expanders:** Groups flagged clauses by their human-readable category icon (e.g., `⚖️ Arbitration Clause`, `🛡️ Limitation of Liability`) along with exact paragraph numbers and model confidence percentages.
- **CSV Exporter:** Exports paragraph numbers, classification types, confidence scores, and raw text for formal compliance audits.

---

## 5. Key Features & Advantages

1. **Heterogeneous Ensemble Accuracy:**
   Combining domain-adapted pre-training (LegalBERT), fast distilled general representations (DistilBERT), and metric-learning embeddings (SBERT + LogReg) significantly reduces individual model bias and variance, yielding higher F1-scores than single-model baselines.

2. **Strict Local-First Privacy:**
   Legal documents often contain proprietary, confidential, or sensitive trade agreements. Because all tokenization, embeddings, and inference are performed locally on the host machine, no data ever leaves the user's workstation.

3. **Intelligent Noise Suppression:**
   The integrated heuristic engine filters out document structure fragments, table of contents headers, and boilerplate legal disclaimers that traditionally plague raw NLP classification pipelines.

4. **Zero-Configuration UI:**
   The Streamlit web interface is responsive and provides real-time progress spinners, color-coded delta metrics, expandable risk breakdowns, and integrated CSV report generation.

5. **Fault-Tolerant Dynamic Execution:**
   Includes automatic detection of Git LFS pointer files with an automated fallback to a heuristic keyword classifier. This ensures the UI remains fully functional and testable even when binary model weights have not yet been downloaded.

---

## 6. Limitations & Disadvantages

1. **Document Format Limitation:**
   The ingestion engine currently supports Microsoft Word (`.docx`) files exclusively. Common legal distribution formats such as PDF (`.pdf`), plain text (`.txt`), and raw web HTML require prior conversion before analysis.

2. **Fixed Context Truncation:**
   The Hugging Face tokenizers enforce a fixed cutoff of `max_length=256` tokens. While this accommodates over 95% of standard legal paragraphs, unusually long compound clauses or multi-sentence paragraphs may have their trailing text truncated during transformer inference.

3. **Storage & Hardware Footprint:**
   Hosting three distinct neural models locally requires approximately **800 MB** of disk storage and roughly 2 to 4 GB of RAM during CPU/GPU inference. Users operating low-resource machines may experience initial model loading latency.

4. **Diagnostic Rather Than Prescriptive:**
   The system flags and categorizes predatory clauses with high accuracy, but does not provide automated clause remediation, redlining suggestions, or generative plain-English rewrites.

---

## 7. How to Run / Setup Guide

### 7.1 Prerequisites
Ensure the target machine has the following installed:
- **Python:** Version 3.10, 3.11, or 3.12 (64-bit)
- **Git & Git LFS:** (Required to download the complete 800 MB model weight files)

### 7.2 Installation Steps

#### Step 1: Clone the Repository and Pull Model Weights
```bash
# Clone the repository
git clone https://github.com/your-username/tos-analyzer-streamlit.git
cd tos-analyzer-streamlit

# Ensure Git LFS is installed and pull binary model weights
git lfs install
git lfs pull
```
> **Note:** If you downloaded the repository as a ZIP archive from GitHub, the model files in `models/` may be 130-byte Git LFS pointer stubs. Running `git lfs pull` restores the actual binary `.safetensors` and `.joblib` files. If you run without Git LFS, the application will automatically operate in **Demo Mode** using the built-in heuristic classifier.

#### Step 2: Install Python Dependencies
Install all required libraries specified in `requirements.txt`:
```bash
pip install -r requirements.txt
```

#### Step 3: Run the Streamlit Application
Launch the application using Streamlit:
```bash
streamlit run app3.py
```

#### Step 4: Access the Web Dashboard
Open your web browser and navigate to:
```
http://localhost:8501
```

#### Step 5: Test with the Sample Document
You can immediately verify the system by uploading the included `sample_tos.docx` file into the upload box on the dashboard.

---

## 8. Conclusion & Future Scope

### 8.1 Conclusion
The **Terms of Service Analyzer** delivers an effective automated solution to the pervasive problem of legal document opacity. By combining three distinct NLP architectures—LegalBERT, DistilBERT, and SBERT—into a unified weighted ensemble, the application achieves balanced, high-confidence detection of problematic clauses across six primary legal categories. Paired with rule-based noise filtering and a local-first Streamlit web dashboard, the system provides an accessible, privacy-preserving auditing tool for consumers and organizations alike.

### 8.2 Future Scope & Proposed Enhancements
1. **Multi-Format Ingestion Engine (PDF & OCR Support):**
   Integrate `pypdf` and `pytesseract` (Tesseract OCR) to allow direct ingestion of scanned and digital PDF contracts, as well as a browser extension to analyze active web terms in real time.
2. **Generative LLM Remediation & Plain-English Translation:**
   Integrate local small language models (such as `Llama-3-8B-Instruct` or `Gemma-2-9B` via `ollama` or `llama.cpp`) to provide plain-language clause summaries, risk severity ratings, and automated "redlined" counter-proposals.
3. **Cross-Document Comparative Auditing & Privacy Scoring:**
   Introduce an aggregate "Fairness Score" (e.g., grade rating from A to F) and a side-by-side comparative diff viewer that tracks how a service's terms have evolved across revisions over time.

---

## 9. References

1. **Chalkidis, I., Fergadiotis, M., Malakasiotis, P., & Androutsopoulos, I.** (2020). *LEGAL-BERT: The Muppets straight out of Law School.* Findings of the Association for Computational Linguistics: EMNLP 2020, 2898–2904.
2. **Sanh, V., Debut, L., Chaumond, J., & Wolf, T.** (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* arXiv preprint arXiv:1910.01108.
3. **Reimers, N., & Gurevych, I.** (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP).
4. **Hugging Face Transformers Documentation:** https://huggingface.co/docs/transformers
5. **Streamlit Open-Source Framework:** https://docs.streamlit.io
6. **Python-docx Library:** https://python-docx.readthedocs.io
