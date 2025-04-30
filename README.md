# 🌍 AI-Drive Multilingual Toxicity Detection Tool

This is a multilingual document-level toxicity detection tool designed to analyze texts in **six UN languages** and classify them into toxicity levels using multiple fine-tuned XLM-RoBERTa models. Additionally, it incorporates GPT-4 for arbitration and severity rating across four toxicity dimensions: **bias**, **xenophobia**, **misinformation**, and **cultural superiority**.

## 🔧 Features

- 📂 Upload `.docx` or `.pdf` documents for batch analysis
- 🌐 Handles multilingual inputs (Arabic, Chinese, English, French, Russian, Spanish)
- 🧠 Supports two analysis directions:
  - **Direction A**: 4-class classification (`None`, `Mild`, `High`, `Max`) with multi-model review
  - **Direction B**: 3-class classification of human-labeled data
- 🤖 GPT-4 integration for post-analysis toxicity interpretation
- 🌟 Star-rating system for severity display (0–10 scale)

---

## 🧪 Detection Logic

### Direction A: 4-Class Document Analysis
- **Primary Model**: XLM-R fine-tuned 4-class classifier
- **High/Max Recheck**: Re-evaluates borderline `High/Max` cases with a second model
- **Mild Detection Review**: Verifies borderline `Mild` predictions via a dedicated model
- **Final Output**: Majority-vote combined with confidence thresholding

### Direction B: 3-Class Human-Labeled Analysis
- Uses an independent XLM-R model trained on manually labeled datasets (None, High, Max)

---

## 🔍 GPT Arbitration Layer

After model classification, the input is re-analyzed by GPT-4 using a rubric across:
- Bias
- Xenophobia
- Misinformation
- Cultural Superiority

Each category is scored from `0–10`, visualized via a 5-star rating system.

---

## 🚀 How to Run

> ⚠️ Requires Python ≥ 3.9 and GPU-enabled PyTorch (optional but recommended).

### 1. Activate Conda Environment

```bash
conda activate toxic-pure
