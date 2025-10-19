# 🧠 NIDS-ML: AI-Powered Network Intrusion Detection with Explainability

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange?logo=tensorflow)
![LightGBM](https://img.shields.io/badge/LightGBM-4.5-brightgreen?logo=lightgbm)
![MITRE ATT&CK](https://img.shields.io/badge/MITRE-ATT%26CK-critical?logo=mitre)
![Zeek](https://img.shields.io/badge/Zeek-6.0.3-lightgrey?logo=zeek)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🚀 Overview

**NIDS-ML** is a production-grade **AI-powered Network Intrusion Detection System** (NIDS) capable of detecting and explaining cyberattacks using machine learning and large language models (LLMs).  
It integrates **LightGBM**, **Autoencoder anomaly detection**, and **rule-based logic** into a 3‑tier detection pipeline — enhanced with **Retrieval-Augmented Generation (RAG)** for natural‑language threat explanations mapped to **MITRE ATT&CK** techniques.

> 📈 **Detection Accuracy:** 99.89%  🧠 **LLM Explanation Quality:** 100% complete sentences (no hallucinations)  
> 🧩 **Architecture:** 3‑tier Ensemble  🔍 **Explainability:** RAG + Llama 3.1 (8 B) via Ollama

---

## 🧱 System Architecture (High‑Level)

```
                 ┌────────────────────────────────────────┐
                 │          ZEEK SENSOR (log source)       │
                 └───────────────┬────────────────────────┘
                                 │ conn.log
                                 ▼
                     ┌────────────────────────────┐
                     │   Feature Engineering (77) │
                     └──────────────┬─────────────┘
                                    ▼
                     ┌────────────────────────────┐
                     │  3‑Tier Detection System   │
                     │  • LightGBM                │
                     │  • Autoencoder             │
                     │  • Rule‑based Engine       │
                     └──────────────┬─────────────┘
                                    ▼
                     ┌────────────────────────────┐
                     │     RAG Explainer (LLM)    │
                     │  MITRE ATT&CK Mapping +    │
                     │  Natural‑Language Output    │
                     └──────────────┬─────────────┘
                                    ▼
                          CSV / SIEM Integration
```

*(Add your architecture diagram image here: `/outputs/architecture.png`)*

---

## ⚡ Quickstart Summary

### 1️⃣ Setup Environment
```bash
git clone <your-repo-url> nids-ml && cd nids-ml
conda create -n nids-ml python=3.12 -y
conda activate nids-ml
pip install -r requirements.txt
```

### 2️⃣ Prepare Data
- Place **CIC‑IDS2017** dataset in `data/raw/`
- Run preprocessing:
```bash
python scripts/01_build_feature_space.py --input data/raw --output data/processed
```

### 3️⃣ Train Models
```bash
python scripts/03_train_lightgbm.py
python scripts/03b_train_autoencoder.py
```

### 4️⃣ Build RAG Explainer
```bash
ollama serve &
ollama pull llama3.1:8b
python scripts/04_build_mitre_kb.py --kb-dir kb/mitre --rag-store rag
```

### 5️⃣ Run Live Detection + Explanations
```bash
python scripts/05_integration_test.py --zeek-log zeek/logs/conn.log --out-csv outputs/detections_test.csv
```

---

## 📂 Project Parts

| Part | Description | Link |
|------|--------------|------|
| **Part 1 – Foundation & Early Implementation** | Project scope, architecture, and feature engineering | [📄 Open Part 1](Formatted_Docs/PART_1_FOUNDATION_and_EARLY_IMPLEMENT.md) |
| **Part 2 – Core Implementation & Deployment** | Model training (LightGBM + Autoencoder) and evaluation | [📄 Open Part 2](Formatted_Docs/PART_2_CORE_IMPLEMENTATION_and_DEPLOY.md) |
| **Part 3 – Results, Analysis & Professional Development** | Testing, simulation, explanation metrics & career takeaways | [📄 Open Part 3](Formatted_Docs/PART_3_RESULTS_ANALYSIS_and_PROFESSI.md) |

---

## 📊 Key Results

| Metric | Value |
|---------|-------|
| **LightGBM Accuracy** | 99.89 % |
| **ROC‑AUC** | 1.000  |
| **Autoencoder F1‑Score** | 0.911  |
| **LLM Explanation Completeness** | 100 % (no truncation / tags) |
| **MITRE Technique Coverage** | 47 techniques across 5 tactics |

---

## 🧩 Tech Stack

| Layer | Tools / Frameworks |
|-------|--------------------|
| **ML / AI** | Python · scikit‑learn · LightGBM · TensorFlow/Keras · SMOTE |
| **Explainability** | Ollama · Llama 3.1 (8 B) · ChromaDB · Sentence‑Transformers |
| **Network** | Zeek 6.0.3 · pfSense VM Lab |
| **Visualization** | Matplotlib · Seaborn · CSV (SIEM ready) |
| **Infra & Dev** | Jupyter · VS Code · Git · Linux/Windows VMs |

---

## 🎓 Professional Context

**Skills Demonstrated**
- Ensemble ML design, deep learning autoencoder training  
- RAG implementation for AI Explainability  
- Zeek network forensics integration  
- MITRE ATT&CK mapping automation  
- Full pipeline validation & unit testing  

**Career Alignment**
| Level | Role Examples |
|--------|----------------|
| 🎯 Entry | SOC Analyst · ML Ops Intern · Threat Detection Engineer |
| 🧠 Mid | Security Engineer · AI Explainability Developer |
| 🚀 Senior | AI Security Architect · Research Lead (ML for Cybersecurity) |

---

## 👤 Author

**Prepared by [Prageeth Panicker](https://www.linkedin.com/in/prageeth-panicker)**  
Cybersecurity · AI Automation · Risk Management  

> “Combining machine learning precision with human‑readable explanations for trustworthy network defense.”

---

## 📜 License

MIT License © 2025 Prageeth Panicker · All Rights Reserved
