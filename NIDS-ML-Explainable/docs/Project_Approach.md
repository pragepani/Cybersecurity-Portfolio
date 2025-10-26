# Project Approach: AI-Driven Network Intrusion Detection with Explainability

## 1. Objective

This project aims to develop an AI-driven Network Intrusion Detection System (NIDS) that integrates classical machine-learning models, unsupervised anomaly detection, and rule-based validation into a three-tier detection framework, enhanced by a RAG-based LLM Explainer for interpretability and MITRE ATT&CK alignment.

The system ingests Zeek network logs in real time and classifies events with near-zero false negatives while providing context-rich, explainable insights.

---

## 2. Architectural Overview

| Layer | Purpose | Core Components |
|-------|---------|-----------------|
| **Data & Preprocessing** | Collects, cleans, and standardizes Zeek network traffic. | Zeek logs → Pandas/Parquet conversion → Label encoding → Feature scaling |
| **Detection Layer (3-Tier Ensemble)** | Multi-model ensemble for robust classification and anomaly detection. | ① LightGBM (Supervised) ② Autoencoder (Unsupervised) ③ Rule-Based Engine |
| **Explanation Layer (3-Tier RAG)** | Generates contextual attack explanations and MITRE mappings. | ① RAG + **Llama 3.1:8b LLM** ② Hybrid Templates ③ MITRE ATT&CK JSON mappings |
| **Integration Layer** | Enables live detection and feedback loop from Zeek to dashboard. | Real-time log ingestion → Inference → Output CSV/Alerts |

### Workflow Summary

1. **Zeek** exports connection, DNS, and HTTP logs.
2. Data pipeline parses TSV → cleans → converts to Parquet.
3. **Tier-1 LightGBM** performs supervised attack classification.
4. **Tier-2 Autoencoder** validates anomalies via reconstruction error.
5. **Tier-3 Rule Engine** cross-verifies known signatures/patterns.
6. **Hybrid RAG Explainer** interprets the event, links to MITRE ATT&CK, and caches for rapid recall (first occurrence uses LLM, repeated detections use templates).

---

## 3. Methodology Highlights

### Part 1 – Data Preparation (File 01)
- Extracted relevant features from Zeek-formatted datasets.
- Handled missing, skewed, and imbalanced data via SMOTE oversampling.
- Derived 77 bidirectional flow features for session-level classification.

### Part 2 – Model Training (File 02)
- Trained LightGBM for high-speed supervised classification (AUC ≈ 0.9999).
- Implemented Autoencoder for anomaly detection on unseen traffic.
- Performed 5-fold cross-validation + Friday hold-out validation.

### Part 3 – System Optimization (File 03)
- Tuned detection thresholds for lower false positives.
- Added ensemble voting and caching layer.
- Introduced latency telemetry and model priority upgrade mechanisms.

### Part 4 – Explainability (File 04)
- Deployed a Retrieval-Augmented Generation (RAG) pipeline.
- Integrated **Llama 3.1:8b** (via Ollama) for dynamic explanations.
- Linked LLM responses with MITRE ATT&CK techniques for interpretability.
- Implemented keyword-based retrieval from JSON knowledge base containing 47 MITRE techniques.

### Part 5 – Zeek Integration (File 05)
- Built live-stream interface reading Zeek logs → model pipeline.
- Generated final CSV output (`zeek_3tier_final.csv`) for SOC dashboard.
- Validated ensemble predictions with near-real-time inference.
- **Implemented parallel LLM processing** using ThreadPoolExecutor for 4-8x faster explanation generation.

---

## 4. 3-Tier Detection Logic

| Tier | Engine | Role | Decision Basis |
|------|---------|------|----------------|
| **Tier 1** | LightGBM | Primary classifier | Supervised learning on labeled dataset |
| **Tier 2** | Autoencoder | Anomaly verifier | Reconstruction error threshold |
| **Tier 3** | Rule-Based | Policy check | Known attack patterns (Snort-style) |

**Final Decision:**
```
(Tier 1 == Attack OR Tier 2 > Threshold OR Tier 3 == Match) → Flag as Intrusion
```

---

## 5. Explainability (Hybrid RAG)

| Fallback Level | Method | Purpose |
|----------------|---------|----------|
| **Level 1** | RAG + **Llama 3.1:8b LLM (Ollama)** | Contextual, natural-language reasoning |
| **Level 2** | Hybrid Templates | Cached responses for frequent alerts |
| **Level 3** | MITRE ATT&CK Mapping | Deterministic fallback for offline use |

**The Explainer produces human-readable justifications such as:**

> *"This pattern corresponds to T1110 – Brute Force as multiple failed SSH logins were observed over short intervals."*

### Explanation Generation Process:
1. **Keyword Search**: Query MITRE knowledge base JSON for relevant techniques based on attack type
2. **LLM Generation**: Llama 3.1:8b generates contextual explanation (2-3s per attack)
3. **Smart Caching**: First occurrence of attack type uses LLM, subsequent same-type attacks use cached template (0.05s)
4. **Parallel Processing**: Multiple attacks processed simultaneously using ThreadPoolExecutor (4-8 workers based on CPU threads)

---

## 6. Key Results

| Metric | LightGBM | Autoencoder | Ensemble |
|--------|-----------|-------------|-----------|
| **Accuracy** | 99.89% | 85.58% | ≈ 99.9% |
| **ROC-AUC** | 0.9999 | 0.91 | 0.999+ |
| **False Positive Rate** | < 0.3% | Moderate | Reduced via voting |
| **Detection Speed** | High (ms range) | Moderate | Optimized via caching |

### Live Zeek Detection Performance:
- **Detection Rate**: 77-99% (varies with traffic patterns)
- **Throughput**: 2-8 flows/s
- **LLM Generation**: 2-3s per attack (parallel processing)
- **Processing Time**: 15-20s for typical batch (7-16 attacks)

---

## 7. Deployment Summary

**Environment:**
- Python 3.10 · Zeek 5.x · **Ollama + Llama 3.1:8b**
- scikit-learn · LightGBM · Keras · Pandas · NumPy
- ThreadPoolExecutor (Parallel Processing)

**Integration:**
- pfSense / Zeek sensor nodes forward logs → detection VM
- SSH-based log retrieval from Zeek VM
- Real-time feature engineering and 3-tier detection

**Storage:**
- MITRE ATT&CK knowledge base (JSON - 47 techniques)
- Parquet (Processed datasets)
- CSV outputs (`zeek_3tier_final.csv`)

**Optimization:**
- Smart caching for repeated attack types
- Parallel LLM processing (multi-threaded)
- Multi-core CPU utilization for explanation generation
- Keyword-based MITRE technique retrieval for fast lookup

### Future Enhancements:
- [ ] GPU-optimized inference with CUDA
- [ ] REST API or Kafka streaming for SOC pipelines
- [ ] Vector database integration (ChromaDB/Pinecone) for semantic search
- [ ] Extended MITRE coverage via automatic technique expansion
- [ ] Integration with SIEM tools (ELK / Wazuh / OpenCTI)
- [x] Parallel LLM processing ✅ **IMPLEMENTED**

---

## 8. Repository Structure

```
nids-ml/
│
├── data/                     # Raw & processed datasets
├── models/                   # Trained LightGBM / Autoencoder / Thresholds
├── explainer/                # RAG components & MITRE knowledge base
│   ├── mitre_knowledge_base_production.json  # 47 MITRE techniques
│   ├── rag_explainer.py                      # RAG explainer implementation
│   ├── rag_production_config.json            # Configuration
│   ├── telemetry.py                          # Performance monitoring
│   └── __pycache__/
├── notebooks/                # Consolidated Jupyter notebooks (Files 01–05)
│   ├── 01_NIDS_Development_Part1.ipynb
│   ├── 02_NIDS_Development_Part2.ipynb
│   ├── 03_Priority_Upgrade.ipynb
│   ├── 04_RAG_Implementation.ipynb
│   └── 05_Zeek_Integration.ipynb
├── scripts/                  # Automation & inference scripts
│   └── step7_parallel_processing.py
├── results/                  # Detection results and logs
│   └── zeek_3tier_final.csv
└── docs/
    ├── Project_Approach.md   # This document
    └── README.md             # Project overview
```

---

## 9. Technical Specifications

### Detection Models:
- **LightGBM**: 99.89% test accuracy, trained on CIC-IDS2017 dataset (200K samples)
- **Autoencoder**: 85.58% anomaly detection rate, reconstruction error threshold-based
- **Rule Engine**: Pattern matching for SSH-Patator, FTP-Patator, DoS, Port Scan

### Explainability System:
- **Knowledge Base**: JSON file with 47 MITRE ATT&CK techniques
- **Retrieval Method**: Keyword-based lookup by attack type
- **LLM**: Llama 3.1:8b (4.9GB model) via Ollama local inference
- **Cache Strategy**: Attack-type based template caching for 100x speedup on repeats

### Performance Optimization:
- **Parallel Workers**: Dynamic allocation based on CPU threads (`os.cpu_count()`)
- **Thread Safety**: Lock-based progress tracking for concurrent LLM calls
- **Error Handling**: Graceful fallback to templates on LLM failures
- **Fast Lookup**: Direct JSON access for MITRE technique mapping

---

## 10. Validation & Testing

### Dataset Split:
- Training: Monday-Thursday (CIC-IDS2017)
- Test: Random 20% holdout
- Friday Validation: Entire Friday dataset (50,000 samples)

### Cross-Validation:
- 5-fold Stratified K-Fold
- Mean accuracy: 99.88% ± 0.07%

### Live Testing:
- Zeek VM (Ubuntu 22.04) with real network traffic
- Tested on 600 connection logs → 552 flows → 548 attacks detected
- Validated Zeek UID correlation for forensic investigation

---

## 11. Key Innovations

1. **3-Tier Ensemble Architecture**: Combines supervised, unsupervised, and rule-based detection for robust performance
2. **Hybrid RAG Explainer**: Balances LLM generation quality with template caching speed
3. **MITRE ATT&CK Integration**: Automatic technique mapping with keyword-based retrieval
4. **Parallel LLM Processing**: Multi-threaded explanation generation for production scalability
5. **Smart Caching**: Attack-type based caching reduces latency by 100x for repeated attacks
6. **Zeek Integration**: Preserves Zeek UIDs for seamless forensic correlation
7. **Lightweight Design**: No vector database dependency - uses fast JSON lookup

---

## 12. MITRE ATT&CK Coverage

The system currently maps to **47 MITRE ATT&CK techniques** across the following tactics:

**Attack Types Covered:**
- SSH-Patator (T1110.001, T1021.004)
- FTP-Patator (T1110.001, T1071.002)
- DoS / DoS Hulk (T1498, T1498.001)
- Port Scan (T1046)
- Web Attacks (T1190, T1059.007)
- Brute Force (T1110.001)
- Infiltration (T1078, T1133)
- Bot attacks (T1071.001, T1573)

**Retrieval Method:**
```python
# Fast keyword-based lookup
attack_type = "SSH-Patator"
techniques = mitre_kb[attack_type]['techniques']  # Returns: ['T1110.001', 'T1021.004']
```

---

## 13. References

- [Zeek Network Security Monitor](https://zeek.org/)
- [MITRE ATT&CK Framework](https://attack.mitre.org/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Ollama + Llama 3.1:8b](https://ollama.ai/)
- [CIC-IDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)

---

**Document Version:** 2.1  
**Last Updated:** October 2025  
**Author:** Prageeth Panicker
