# 🌍 AI-Drive Multilingual Toxicity Analysis Tool

## Overview

This is a **Streamlit-based web application** designed to detect and analyze **multilingual toxic content** in text and documents. The tool supports `.docx`, `.pdf`, or raw pasted text, and uses a **multi-model architecture** combining **XLM-RoBERTa classifiers** and **GPT-4 arbitration** to classify content severity.

The system detects **four toxicity levels** (`None`, `Mild`, `High`, `Max`) and evaluates content across key dimensions like **xenophobia**, **bias**, **cultural superiority**, and **misinformation**.

---

## 🔑 Features

- **Multilingual Support**: Works across six UN languages (English, Chinese, Arabic, Russian, French, Spanish)
- **Document Upload**: Supports PDF and Word files
- **Text Input**: Paste any free-form content
- **4-Class + 3-Class Mode Switch**:
  - Direction A: 4-class detection with rechecking logic
  - Direction B: 3-class detection from human-labeled data
- **GPT-4 Arbitration Layer**: Independent post-analysis rating with scores and explanations
- **Star-Rating Output**: Friendly 5-star visualization of severity level (0–10 scale)

---

## 📁 Setup and Installation

### 🔧 Prerequisites

- Python 3.9+
- GPU optional (runs on CPU or CUDA)
- Conda environment recommended (e.g., `toxic-pure`)

### 📦 Installation

```bash
pip install torch transformers streamlit openai python-docx python-dotenv pymupdf

### 🧪 Usage Instructions

 Option 1: Upload a File

1. Click **"Upload a PDF or DOCX file"**
2. Choose a document in one of the supported languages

 Option 2: Paste Text

1. Use the **text input** field to enter a paragraph or article

 Step 3: Choose Detection Mode

- **Direction A** – 4-class classification with review models for:
  - Mild → High
  - High → Max
- **Direction B** – 3-class classification based on human-labeled data

 Step 4: Run Analysis

Click the **Analyze** button to start detection.

Output includes:**
- Final **Toxicity Level**: `None`, `Mild`, `High`, `Max`
- **Scaled Score**: 0–10
- ⭐ **Star Rating**
- 🧠 **GPT-4 Content Analysis** across:
  - Bias
  - Xenophobia
  - Cultural Superiority
  - Misinformation
