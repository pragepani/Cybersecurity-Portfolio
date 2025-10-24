# 🧠 AI-Driven Network Intrusion Detection System (NIDS) with Explainability

**Author:** Prageeth Panicker  
**Focus Areas:** Cybersecurity · AI/ML · Explainable AI · Network Analytics  
**Version:** 2.0 · Last Updated: October 2025  

---

## 🚀 Overview

This project implements a **multi-tier Network Intrusion Detection System (NIDS)** that combines **Machine Learning, Autoencoder-based anomaly detection, and rule-based correlation**, integrated with a **Retrieval-Augmented Generation (RAG)** explainer powered by **Llama 3.1 : 8B (via Ollama)** for **MITRE ATT&CK** mapping and contextual insights.

The solution processes **Zeek network logs** in real time and produces **interpretable security alerts**—bridging the gap between data-driven detection and human-understandable threat intelligence.

---

## 🧩 System Architecture

| Layer | Description | Core Components |
|--------|--------------|----------------|
| **Data Layer** | Zeek network traffic parsing, cleaning, and normalization | Zeek logs → Pandas/Parquet → LabelEncoder → Scaler |
| **Detection Layer (3-Tier Ensemble)** | Combines supervised, unsupervised, and rule-based detection | LightGBM · Autoencoder · Rule Engine |
| **Explanation Layer (3-Tier RAG)** | Provides contextual, human-readable insights | **Llama 3.1 : 8B (Ollama)** · Hybrid Templates · MITRE Mapping |
| **Integration Layer** | Real-time ingestion and feedback loop | Zeek → Inference → Output CSV / Alerts |

---

## ⚙️ Project Workflow

1. **Zeek** exports connection, HTTP, and DNS logs.  
2. Data is preprocessed, normalized, and stored in **Parquet format**.  
3. **LightGBM** performs primary supervised classification.  
4. **Autoencoder** validates anomalies using reconstruction error thresholds.  
5. **Rule-based engine** confirms matches with known signatures.  
6. **RAG Explainer** interprets alerts using MITRE ATT&CK and **Llama 3.1 : 8B LLM** (*with smart caching – first occurrence uses LLM, repeated detections use templates*).  
7. Results are exported to **`zeek_3tier_final.csv`** and SOC dashboards.

---

## 🧠 Model Design (3-Tier Detection Logic)

| Tier | Engine | Role | Decision Basis |
|------|---------|------|----------------|
| **Tier 1** | LightGBM | Supervised detection | Trained on labeled CIC-IDS2017 dataset |
| **Tier 2** | Autoencoder | Anomaly detection | Reconstruction error threshold |
| **Tier 3** | Rule Engine | Policy-level validation | Snort/Zeek-style signature rules |

> **Final decision logic:**  
> `(Tier 1 == Attack) OR (Tier 2 > Threshold) OR (Tier 3 == Match) → Intrusion = True`

---

## 💬 Explainability (Hybrid RAG Explainer)

| Fallback Level | Method | Purpose |
|----------------|---------|----------|
| **Level 1** | RAG + Llama 3.1 : 8B (Ollama) | Dynamic, context-rich explanations |
| **Level 2** | Hybrid Templates | Fast cached response |
| **Level 3** | MITRE ATT&CK JSON | Deterministic fallback |

**Example Output:**  
> *"Detected behavior aligns with MITRE Technique T1110 – Brute Force, showing multiple failed SSH logins from a single IP within a short interval."*

---

## 📈 Key Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **99.89 %** |
| ROC–AUC | **1.0000** |
| Detection Rate (Live Zeek) | **77 – 99 %** |
| LLM Generation Speed | **2 – 3 s per attack (parallel)** |
| Throughput | **2 – 8 flows/s** |

---

## 🧰 Tech Stack

**Languages:** Python 3.10  
**Libraries:** scikit-learn · LightGBM · Keras · Pandas · NumPy  
**Explainability:** **Ollama (local inference)** · **Llama 3.1 : 8B**  
**MITRE Mapping:** Keyword-based retrieval with JSON knowledge base  
**Optimization:** Parallel LLM processing (ThreadPoolExecutor) · Multi-core utilization  
**Network Sensor:** Zeek 5.x  
**Visualization:** Matplotlib · Plotly  
**Environment:** pfSense · Ubuntu 22.04 VM  

---

## 🗂️ Repository Structure

```text
nids-ml/
|
|-- data/                    # Raw & processed datasets
|-- models/                  # Trained LightGBM / Autoencoder / Thresholds
|-- explainer/               # RAG components & MITRE knowledge base
|   |-- mitre_knowledge_base_production.json
|   |-- rag_explainer.py
|   |-- rag_production_config.json
|   |-- telemetry.py
|   \-- __pycache__/
|-- notebooks/               # Consolidated Jupyter notebooks (Files 01–05)
|-- scripts/                 # Automation & inference scripts
|-- results/                 # Detection results and Zeek logs
\-- docs/
    \-- Project_Approach.md  # Technical explanation document
```

---

## 🧪 Workflow Summary by Notebook

| File | Purpose |
|------|----------|
| **01 – NIDS Development Part 1** | Data ingestion, preprocessing, feature creation |
| **02 – NIDS Development Part 2** | Model training (LightGBM + Autoencoder) |
| **03 – Priority Upgrade** | Optimization, latency reduction, ensemble tuning |
| **04 – RAG Implementation** | Explainability using Llama 3.1 : 8B LLM + MITRE mapping |
| **05 – Zeek Integration** | Real-time log ingestion, final CSV output |
| **zeek_3tier_final.csv** | Final detection dataset |

---

## 🔮 Future Enhancements

- [x] Parallel LLM processing for faster explanation generation ✅ DONE  
- [ ] GPU-accelerated inference with CUDA 13  
- [ ] API-based live detection service (FastAPI / Kafka)  
- [ ] Vector database integration (ChromaDB) for semantic MITRE search  
- [ ] Advanced caching and telemetry dashboards  
- [ ] Automatic MITRE ATT&CK coverage expansion  
- [ ] Integration with SIEM tools (ELK / Wazuh / OpenCTI)

---

## 📚 References

- [Zeek Network Security Monitor](https://zeek.org/)  
- [MITRE ATT&CK Framework](https://attack.mitre.org/)  
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)  
- [Ollama + Llama 3.1 : 8B](https://ollama.ai/)

---

## 👨‍💻 Author

**Prageeth Panicker**  
Cybersecurity | AI Automation | Risk Management  
🔗 [LinkedIn](https://www.linkedin.com/in/prageeth-panicker) · [GitHub](https://github.com/pragepani)

---

### 📄 License

This project is licensed under the **MIT License** – you are free to use, modify, and distribute this code for both personal and commercial purposes, provided that proper credit is given.

```
MIT License

Copyright (c) 2025 Prageeth Panicker

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
