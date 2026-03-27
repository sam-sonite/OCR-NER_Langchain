# 📄 OCR + NER Evaluation Pipeline (FUNSD)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![LangChain](https://img.shields.io/badge/LangChain-Orchestration-blueviolet)
![Model](https://img.shields.io/badge/Model-LayoutLMv3-informational)
![Benchmark](https://img.shields.io/badge/Benchmark-FUNSD-purple)
![Evaluation](https://img.shields.io/badge/Focus-Evaluation%20Framework-critical)
![Metrics](https://img.shields.io/badge/Metrics-CER%20%7C%20WER%20%7C%20F1-blue)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)

---

## 📌 Overview

This repository presents an **evaluation-focused document understanding pipeline** built on the FUNSD dataset. The system is designed to measure the performance of document AI models across OCR and Named Entity Recognition (NER) tasks, with a strong emphasis on **evaluation metrics, robustness, and downstream impact**.

The pipeline simulates an end-to-end OCR + NER workflow while isolating model performance for controlled benchmarking.

---

## 🧠 How It Works

The system operates in two stages:

### 1. OCR (Simulated)
The OCR stage is **not executed using a vision model**. Instead, the pipeline directly uses ground truth text provided by the FUNSD dataset.

- Acts as a **perfect OCR baseline**
- Eliminates transcription errors
- Allows focused evaluation of downstream tasks

### 2. NER (Entity Extraction)
The extracted text is passed to a **layout-aware transformer model**:

- Model: `nielsr/layoutlmv3-finetuned-funsd`
- Input: text + bounding boxes + document image
- Output: labeled entities (e.g., keys, values, headers)

# OCR-NER_Langchain<img width="1024" height="1536" alt="ChatGPT Image Mar 27, 2026, 06_00_15 PM" src="https://github.com/user-attachments/assets/5333285e-b3e2-4175-b6c8-428414eb0a96" />
---

