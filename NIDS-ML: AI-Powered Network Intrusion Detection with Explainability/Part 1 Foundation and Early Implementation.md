PART 1: FOUNDATION & EARLY IMPLEMENTATION
# Advanced Network Intrusion Detection System with AI-Powered Explainability
## A Production Implementation of Multi-Tier Detection with RAG-Enhanced MITRE ATT&CK Mapping

**Complete Implementation Guide**

---

## Document Information

**Project Title:** Advanced NIDS with AI Explainability using RAG and MITRE ATT&CK Framework  
**Author:** Prageeth Panicker  
**Date:** October 18, 2025  
**Version:** 1.0  
**Status:** Production-Ready Implementation  

---

## Executive Summary

### Project Challenge

Deploy a production-grade Network Intrusion Detection System (NIDS) capable of not only detecting cyber attacks with high accuracy but also providing human-interpretable explanations grounded in the MITRE ATT&CK framework. The challenge encompassed processing 2.3 million network flow records, engineering 77 discriminative features, training ensemble detection models, and integrating large language models (Llama3.1:8b) with retrieval-augmented generation (RAG) to produce natural language threat intelligence reports.

### Solution Implemented

Successfully deployed a 3-tier hybrid detection system combining:
- **Tier 1**: LightGBM gradient boosting (99.89% accuracy)
- **Tier 2**: Autoencoder-based anomaly detection (unsupervised learning)
- **Tier 3**: Rule-based pattern matching (expert knowledge)

Integrated with production RAG explainer system featuring:
- Semantic search over 47 MITRE ATT&CK techniques using ChromaDB vector database
- Llama3.1:8b LLM for natural language explanation generation
- Deterministic technique mapping preventing hallucinations
- Real-time Zeek log processing with UID correlation

### Key Outcomes

**Detection Performance:**
- Test Set Accuracy: 99.89%
- Test ROC-AUC: 1.0000
- Friday Hold-out Accuracy: 99.798%
- Real-time Detection Rate: 37.5% (live benign traffic) to 80-100% (active attacks)

**Explainability Achievements:**
- 47 MITRE ATT&CK techniques mapped with semantic search
- 66.7% LLM-generated explanations (no thinking tags)
- Zero hallucinated technique IDs (forced from retrieval)
- Complete Zeek UID correlation for forensic investigation

**System Integration:**
- Live Zeek log ingestion and flow aggregation
- Multi-tier ensemble voting with confidence scoring
- CSV export with 20+ fields for SIEM integration
- Alert ID generation with session tracking

### Technical Skills Demonstrated

**Machine Learning & AI:**
- Imbalanced dataset handling (SMOTE oversampling)
- Gradient boosting (LightGBM) hyperparameter tuning
- Autoencoder neural network architecture design
- Cross-validation and hold-out testing methodology
- Feature engineering (77 bidirectional flow features)

**Natural Language Processing:**
- Retrieval-Augmented Generation (RAG) architecture
- Vector database (ChromaDB) with semantic embeddings
- Large Language Model (Llama3.1:8b) integration via Ollama
- Prompt engineering for technical explanations
- Hallucination prevention through deterministic grounding

**Systems Integration:**
- Real-time Zeek log parsing and aggregation
- Multi-tier detection pipeline orchestration
- Production error handling and fallback mechanisms
- CSV export for SIEM platforms
- Zeek UID correlation for forensic workflows

**Cybersecurity Domain:**
- MITRE ATT&CK framework deep integration
- Network flow analysis (bidirectional features)
- Attack taxonomy (DoS, brute force, port scan, infiltration)
- Threat intelligence report generation
- Security Operations Center (SOC) workflow design

### Business Value

**Operational Impact:**
- **Detection Capability**: 99.89% accuracy detecting 14 attack types across Impact, Credential Access, Discovery, Command & Control, and Initial Access tactics
- **Explainability**: Natural language threat intelligence reports referencing MITRE techniques, reducing analyst investigation time
- **Forensic Correlation**: Zeek UID integration enables immediate drill-down to full packet capture and protocol logs
- **SIEM Integration**: Structured CSV output with 20+ fields ready for Splunk, ELK, or Wazuh ingestion
- **Scalability**: Cluster-ready architecture supporting additional detection tiers and data sources

**Risk Reduction:**
- Early detection of SSH brute force (T1110.001) before account compromise
- Port scan detection (T1046) revealing reconnaissance phase
- DoS attack identification (T1498) protecting availability
- Behavioral anomaly detection catching zero-day attacks missed by signatures

**Cost Efficiency:**
- Open-source technology stack (no licensing costs)
- Automated explanation generation (reducing analyst manual work)
- Reusable pipeline for additional datasets (Friday-WorkingHours, Monday-Monday, etc.)
- Horizontal scaling without vendor lock-in

---

## Table of Contents

### 1. Project Overview
   1.1 What is a Network Intrusion Detection System?  
   1.2 Why AI Explainability Matters  
   1.3 Original Specification vs Actual Implementation  
   1.4 Technology Stack Overview  

### 2. System Architecture
   2.1 High-Level Architecture Diagram  
   2.2 Data Flow Pipeline  
   2.3 Component Interactions  
   2.4 Network Topology  

### 3. Scope and Objectives
   3.1 Project Scope  
   3.2 Primary Objectives  
   3.3 Learning Outcomes  
   3.4 Out of Scope  

### 4. Prerequisites
   4.1 Infrastructure Requirements  
   4.2 Software Dependencies  
   4.3 Dataset Requirements  
   4.4 Knowledge Prerequisites  

### 5. Implementation: File 01 - Dataset Preparation
   5.1 Dataset Overview (CIC-IDS2017)  
   5.2 Exploratory Data Analysis  
   5.3 Data Quality Assessment  
   5.4 Class Distribution Analysis  
   5.5 Feature Space Exploration  

### 6. Implementation: File 02 - Feature Engineering
   6.1 Feature Engineering Strategy  
   6.2 Bidirectional Flow Features  
   6.3 Statistical Aggregations  
   6.4 Feature Selection and Validation  
   6.5 Data Preprocessing Pipeline  

### 7. Implementation: File 03 - Model Training
   7.1 Training Strategy  
   7.2 LightGBM Configuration  
   7.3 Autoencoder Architecture  
   7.4 Cross-Validation Results  
   7.5 Friday Hold-out Testing  

### 8. Implementation: File 04 - RAG Explainer
   8.1 MITRE ATT&CK Knowledge Base  
   8.2 Vector Database Setup (ChromaDB)  
   8.3 Semantic Search Implementation  
   8.4 Llama3.1:8b Integration  
   8.5 Hallucination Prevention  

### 9. Implementation: File 05 - Live Detection
   9.1 3-Tier Detection Architecture  
   9.2 Zeek Log Integration  
   9.3 Flow Aggregation Logic  
   9.4 Ensemble Voting Mechanism  
   9.5 Explanation Generation Workflow  
   9.6 CSV Export and SIEM Integration  

### 10. Testing and Validation
   10.1 Unit Testing Strategy  
   10.2 Integration Testing  
   10.3 Live Attack Simulation  
   10.4 Explanation Quality Assessment  

### 11. Architectural Challenges & Solutions
   11.1 Imbalanced Dataset Handling  
   11.2 Explanation Truncation Issue  
   11.3 LLM Thinking Tags Problem  
   11.4 Zeek UID Correlation  
   11.5 Feature Name Warnings  

### 12. Results and Outcomes
   12.1 Detection Performance Metrics  
   12.2 Explanation Quality Metrics  
   12.3 System Integration Results  
   12.4 Comparison to Baseline  

### 13. Skills
   13.1 Technical Skills Matrix    

### 14. Lessons Learned
   14.1 Technical Lessons  
   14.2 Architectural Lessons  
   14.3 Process Lessons  

### 15. Future Roadmap
   15.1 Immediate Enhancements (0-2 weeks)  
   15.2 Intermediate Expansion (2-3 months)  
   15.3 Long-term Integration (3-6 months)  

### 16. Conclusion
   16.1 Project Summary  
   16.2 Key Takeaways  
   16.3 Final Thoughts  

### 17. References
   17.1 Academic Papers  
   17.2 Technical Documentation  
   17.3 Tools and Frameworks  

### 18. Appendices
   Appendix A: Complete Feature List (77 Features)  
   Appendix B: MITRE ATT&CK Technique Mappings  
   Appendix C: Sample Detection Reports  
   Appendix D: Configuration Files  

---

## 1. Project Overview

### 1.1 What is a Network Intrusion Detection System?

A Network Intrusion Detection System (NIDS) is a cybersecurity technology that monitors network traffic for suspicious activity and potential threats. Unlike firewalls that block traffic based on rules, NIDS systems analyze traffic patterns to identify attacks that bypass perimeter defenses.

**Traditional NIDS Limitations:**
- Signature-based detection misses zero-day attacks
- High false positive rates requiring manual investigation
- No context on WHY an alert was generated
- Limited integration with threat intelligence frameworks
- Black-box decision making without explanations

**Modern AI-Enhanced NIDS:**
- Machine learning models detect behavioral anomalies
- Ensemble methods combine multiple detection approaches
- Natural language explanations reference MITRE ATT&CK
- Automated threat intelligence report generation
- Integration with SIEM platforms for correlation

### 1.2 Why AI Explainability Matters

**The Challenge:**
Modern machine learning models achieve high accuracy but operate as "black boxes" - security analysts receive alerts without understanding the reasoning behind detections.

**Real-World Impact:**
```
Traditional Alert:
┌─────────────────────────────────────┐
│ ALERT: Attack Detected              │
│ Source: 192.168.10.100              │
│ Confidence: 98%                     │
│ Action: Investigate                 │
└─────────────────────────────────────┘
```

**Analyst Questions (Unanswered):**
- What type of attack is this?
- What MITRE ATT&CK techniques are involved?
- What network behavior triggered the detection?
- What immediate actions should I take?
- How do I correlate this with Zeek logs?

**AI-Explained Alert:**
```
┌─────────────────────────────────────────────────────────────────┐
│ 🚨 SSH-PATATOR ATTACK DETECTED                                 │
├─────────────────────────────────────────────────────────────────┤
│ Alert ID: NIDS-20251018-142345-0001                            │
│ Zeek UID: CYwKzH3VrF9P1d2H5a                                   │
│ Confidence: 100.0% (Autoencoder + Rule-SSH)                    │
│                                                                 │
│ 🎯 MITRE ATT&CK:                                               │
│    T1110.001 (Password Guessing)                               │
│    T1021.004 (Remote Services: SSH)                            │
│                                                                 │
│ 💡 EXPLANATION:                                                │
│    This attack is attempting to brute-force guess passwords    │
│    on remote systems using SSH (T1021.004), systematically     │
│    guessing passwords to attempt access to accounts            │
│    (T1110.001). The detection was triggered by a high rate     │
│    of SYN and RST flags (16 SYN, 10 RST), indicating an       │
│    automated login attempt with a failed login ratio           │
│    exceeding 80% (>10/sec authentication rate).                │
│                                                                 │
│ 🛡️ RECOMMENDED ACTION:                                         │
│    Block the IP address associated with this attack and        │
│    review SSH configuration for enhanced password policies     │
│    and brute-force protection mechanisms.                      │
│                                                                 │
│ 📋 ZEEK CORRELATION:                                           │
│    grep 'CYwKzH3VrF9P1d2H5a' /opt/zeek/logs/current/conn.log  │
└─────────────────────────────────────────────────────────────────┘
```

**Benefits:**
- **Faster Triage**: Analyst immediately understands attack type
- **Contextualized Decisions**: MITRE techniques guide response
- **Forensic Correlation**: Zeek UID enables immediate log drill-down
- **Training Value**: Junior analysts learn attack patterns
- **Audit Trail**: Documented reasoning for security incidents

### 1.3 Original Specification vs Actual Implementation

**Original Vision:**
- Deploy NIDS on network monitoring infrastructure
- Achieve high detection accuracy (>95%)
- Generate basic alerts for security events
- Export logs for manual analysis

**Actual Implementation - Enhanced Scope:**

| Aspect | Original Plan | Actual Implementation | Enhancement |
|--------|--------------|----------------------|-------------|
| **Detection** | Single model | 3-tier ensemble (LightGBM + Autoencoder + Rules) | Multi-model consensus |
| **Accuracy** | >95% target | 99.89% achieved | Exceeded by 4.89% |
| **Explainability** | Basic labels | RAG + LLM natural language explanations | Full MITRE context |
| **Integration** | CSV export | Zeek UID correlation + SIEM-ready format | Forensic workflow |
| **Knowledge Base** | None | 47 MITRE ATT&CK techniques with semantic search | Threat intelligence |
| **Attack Types** | 5-10 | 14 attack categories across 5 MITRE tactics | Comprehensive coverage |
| **LLM Integration** | Not planned | Llama3.1:8b with hallucination prevention | Production-grade AI |
| **Deployment** | Batch processing | Real-time Zeek log ingestion | Live monitoring |

**Key Innovations Beyond Specification:**

1. **Retrieval-Augmented Generation (RAG)**
   - Vector database (ChromaDB) with semantic embeddings
   - Hybrid retrieval (semantic + keyword search)
   - Deterministic MITRE technique mapping (zero hallucinations)

2. **Multi-Tier Detection Architecture**
   - Ensemble voting across 3 detection methods
   - Tier-specific confidence scoring
   - Graceful degradation with fallback mechanisms

3. **Production-Grade Explainability**
   - Template fallback when LLM unavailable
   - Flow feature integration in explanations
   - Complete vs truncated explanation resolution
   - Removal of LLM "thinking process" artifacts

4. **Forensic Integration**
   - Zeek UID correlation for packet-level investigation
   - Alert ID generation for incident tracking
   - Timestamp preservation for timeline reconstruction
   - Source/destination port logging for service identification

### 1.4 Technology Stack Overview

**Core Technologies:**
```
┌─────────────────────────────────────────────────────────────┐
│                    TECHNOLOGY STACK                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📊 DATA PROCESSING                                         │
│     ├─ Python 3.12                                          │
│     ├─ Pandas 2.x (data manipulation)                       │
│     ├─ NumPy 1.26 (numerical operations)                    │
│     └─ Scikit-learn 1.5 (preprocessing)                     │
│                                                              │
│  🤖 MACHINE LEARNING                                        │
│     ├─ LightGBM 4.5 (gradient boosting)                     │
│     ├─ TensorFlow/Keras 2.17 (autoencoder)                  │
│     ├─ Imbalanced-learn (SMOTE)                             │
│     └─ Joblib (model persistence)                           │
│                                                              │
│  🧠 AI EXPLAINABILITY                                       │
│     ├─ Ollama (LLM runtime)                                 │
│     ├─ Llama3.1:8b (explanation generation)                 │
│     ├─ ChromaDB 0.4 (vector database)                       │
│     ├─ Sentence-Transformers (embeddings)                   │
│     └─ Custom RAG framework                                 │
│                                                              │
│  🌐 NETWORK MONITORING                                      │
│     ├─ Zeek 6.0.3 (network analysis)                        │
│     ├─ Cluster mode (manager + 2 workers)                   │
│     └─ Real-time log ingestion                              │
│                                                              │
│  📈 VISUALIZATION & REPORTING                               │
│     ├─ Matplotlib/Seaborn (EDA)                             │
│     ├─ Confusion matrices                                   │
│     └─ CSV export (SIEM integration)                        │
│                                                              │
│  🔧 DEVELOPMENT TOOLS                                       │
│     ├─ Jupyter Notebook (interactive development)           │
│     ├─ Git (version control)                                │
│     └─ VSCode/PyCharm (IDE)                                 │
└─────────────────────────────────────────────────────────────┘
```

**Version Details:**
- Python: 3.12.3
- LightGBM: 4.5.0
- TensorFlow: 2.17.0
- ChromaDB: 0.4.x
- Sentence-Transformers: all-MiniLM-L6-v2
- Ollama: Latest (Llama3.1:8b model)

**Infrastructure:**
- OS: Windows 11 (development), Ubuntu 22.04 (Zeek VM)
- RAM: 16GB minimum (32GB recommended)
- Storage: 100GB+ for datasets and models
- Network: VirtualBox internal networks (192.168.30.0/24, 192.168.40.0/24)

---

## 2. System Architecture

### 2.1 High-Level Architecture Diagram
```
┌─────────────────────────────────────────────────────────────────────────┐
│                   NIDS WITH AI EXPLAINABILITY                           │
│                        SYSTEM ARCHITECTURE                               │
└─────────────────────────────────────────────────────────────────────────┘

                        ┌──────────────────┐
                        │   ZEEK SENSOR    │
                        │  192.168.30.80   │
                        │                  │
                        │  • Worker-1 →    │
                        │    enp0s3        │
                        │  • Worker-2 →    │
                        │    enp0s8        │
                        └────────┬─────────┘
                                 │ conn.log
                                 │ (last 200 connections)
                                 ↓
                   ┌─────────────────────────┐
                   │   LOG AGGREGATION       │
                   │   (File 05 - Step 7)    │
                   │                         │
                   │  • Group by flow 5-tuple│
                   │  • Sum packets/bytes    │
                   │  • Aggregate flags      │
                   └────────┬────────────────┘
                            │ Aggregated flows (8-15)
                            ↓
              ┌────────────────────────────────┐
              │   FEATURE ENGINEERING          │
              │   (77 bidirectional features)  │
              │                                │
              │  • Flow statistics             │
              │  • Packet/byte rates           │
              │  • Flag counts (SYN/RST/ACK)   │
              │  • IAT analysis                │
              └────────┬───────────────────────┘
                       │ Feature vectors (8 × 77)
                       ↓
       ┌───────────────────────────────────────────┐
       │        3-TIER DETECTION SYSTEM            │
       │                                           │
       │  ┌─────────────────────────────────────┐ │
       │  │ TIER 1: LightGBM Classifier         │ │
       │  │  • Accuracy: 99.89%                 │ │
       │  │  • Predicts: Attack/Benign          │ │
       │  │  • Outputs: Confidence score        │ │
       │  └──────────────┬──────────────────────┘ │
       │                 │ Vote 1                  │
       │  ┌──────────────┴──────────────────────┐ │
       │  │ TIER 2: Autoencoder Anomaly         │ │
       │  │  • Unsupervised learning            │ │
       │  │  • Threshold: 0.075                 │ │
       │  │  • Detects: Behavioral anomalies    │ │
       │  └──────────────┬──────────────────────┘ │
       │                 │ Vote 2                  │
       │  ┌──────────────┴──────────────────────┐ │
       │  │ TIER 3: Rule-Based Detection        │ │
       │  │  • Port scan (>20 conns, REJ/S0)    │ │
       │  │  • SSH brute force (>10 to port 22) │ │
       │  │  • FTP brute force (>5 to port 21)  │ │
       │  │  • DoS (>1000 pkt/s)                │ │
       │  └──────────────┬──────────────────────┘ │
       │                 │ Vote 3                  │
       │                 ↓                         │
       │  ┌──────────────────────────────────────┐│
       │  │   ENSEMBLE VOTING                    ││
       │  │   • Combine votes from 3 tiers       ││
       │  │   • Final: Attack if votes > 0       ││
       │  │   • Confidence: max(tier scores)     ││
       │  └──────────────┬───────────────────────┘│
       └─────────────────┼────────────────────────┘
                         │ Detection results
                         │ (attack_type, confidence, method)
                         ↓
         ┌───────────────────────────────────────┐
         │   RAG EXPLAINER SYSTEM                │
         │                                       │
         │  ┌─────────────────────────────────┐ │
         │  │ 1. RETRIEVAL                    │ │
         │  │    • Semantic search (ChromaDB) │ │
         │  │    • Keyword fallback           │ │
         │  │    • Top 2 MITRE techniques     │ │
         │  └──────────────┬──────────────────┘ │
         │                 │ MITRE context        │
         │  ┌──────────────┴──────────────────┐ │
         │  │ 2. PROMPT CONSTRUCTION          │ │
         │  │    • Attack type + confidence   │ │
         │  │    • Flow features (SYN/RST)    │ │
         │  │    • MITRE technique details    │ │
         │  └──────────────┬──────────────────┘ │
         │                 │ Prompt               │
         │  ┌──────────────┴──────────────────┐ │
         │  │ 3. LLM GENERATION               │ │
         │  │    • Llama3.1:8b via Ollama     │ │
         │  │    • Temperature: 0.2           │ │
         │  │    • Max tokens: 1000           │ │
         │  │    • Stop: <think> tags         │ │
         │  └──────────────┬──────────────────┘ │
         │                 │ Raw explanation      │
         │  ┌──────────────┴──────────────────┐ │
         │  │ 4. GROUNDING                    │ │
         │  │    • Force MITRE IDs (no halluc)│ │
         │  │    • Extract mitigation from KB │ │
         │  │    • Clean thinking artifacts   │ │
         │  └──────────────┬──────────────────┘ │
         │                 │ Final explanation    │
         │  ┌──────────────┴──────────────────┐ │
         │  │ 5. FALLBACK                     │ │
         │  │    • Template if LLM fails      │ │
         │  │    • KB-based explanation       │ │
         │  └─────────────────────────────────┘ │
         └───────────────┬───────────────────────┘
                         │ Complete explanation
                         │ (text + MITRE + action)
                         ↓
           ┌─────────────────────────────────┐
           │   THREAT INTELLIGENCE REPORT     │
           │                                  │
           │  • Alert ID                      │
           │  • Zeek UID correlation          │
           │  • MITRE techniques              │
           │  • Natural language explanation  │
           │  • Recommended actions           │
           │  • Severity assessment           │
           └─────────────┬────────────────────┘
                         │
                         ↓
              ┌──────────────────────┐
              │   CSV EXPORT         │
              │   (SIEM Integration) │
              │                      │
              │  20+ fields:         │
              │  • alert_id          │
              │  • zeek_uid          │
              │  • src/dst IPs       │
              │  • attack_type       │
              │  • confidence        │
              │  • explanation       │
              │  • mitre_techniques  │
              │  • recommended       │
              │    _action           │
              └──────────────────────┘
```

### 2.2 Data Flow Pipeline

**Stage 1: Data Ingestion (Zeek → Python)**
```
Zeek conn.log (200 connections)
    ↓
Grep filter (remove headers)
    ↓
Parse TSV format
    ↓
Pandas DataFrame (15-25 connections)
```

**Stage 2: Flow Aggregation**
```
Group by: (src_ip, dst_ip, dst_port, proto, conn_state)
    ↓
Aggregate: sum(packets), sum(bytes), sum(flags), mean(duration)
    ↓
Result: 8-15 unique flows
```

**Stage 3: Feature Engineering**
```
77 features calculated:
    • Flow duration (ms)
    • Packets/bytes per second
    • Forward/backward packet counts
    • Flag counts (SYN, FIN, RST, PSH, ACK, URG)
    • Inter-arrival times (mean, std, min, max)
    • Packet length statistics
    • Bulk transfer rates
    ↓
Feature matrix: (8 × 77)
```

**Stage 4: Detection (3 Tiers)**
```
TIER 1: LightGBM
    Input: Feature matrix
    Output: Binary prediction + probability
    Vote: +1 if prediction == Attack
    
TIER 2: Autoencoder
    Input: Scaled features
    Output: Reconstruction error (MSE)
    Vote: +1 if MSE > threshold (0.075)
    
TIER 3: Rules
    Input: Raw flow data
    Output: Rule match (port scan, brute force, DoS)
    Vote: +1 if any rule triggered
    
    ↓
Ensemble Decision:
    final_prediction = 1 if votes > 0 else 0
    final_confidence = max(scores)
    detection_method = ' + '.join(methods)
```

**Stage 5: Explanation Generation**
```
IF attack detected:
    1. Semantic search: Query ChromaDB with attack_type
    2. Keyword search: Lookup technique_mapping dict
    3. Select top 2 MITRE techniques
    4. Build prompt: attack + flow features + MITRE context
    5. LLM generation: Llama3.1:8b produces explanation
    6. Grounding: Force MITRE IDs from retrieval (no hallucinations)
    7. Fallback: Use template if LLM fails
    ↓
Output: {
    explanation: "SSH-Patator attack is attempting...",
    mitre_techniques: ["T1110.001", "T1021.004"],
    recommended_action: "Block IP, enable fail2ban",
    confidence: "100.0%",
    source: "rag_llm"
}
```

**Stage 6: Report Generation & Export**
```
Combine detection + explanation:
    • Generate alert_id (session-based)
    • Preserve zeek_uid for correlation
    • Extract timestamp, src/dst IPs, ports
    • Format as threat intelligence report
    ↓
Display on console (formatted)
    ↓
Export to CSV (20+ fields for SIEM)
```

### 2.3 Component Interactions

**Component Dependency Map:**
┌─────────────────────────────────────────────────────────────┐
│                   COMPONENT INTERACTIONS                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Dataset (CIC-IDS2017)                                      │
│      │                                                       │
│      ├──> File 01: EDA & Preprocessing                      │
│      │       └──> Cleaned parquet files                     │
│      │                                                       │
│      └──> File 02: Feature Engineering                      │
│            └──> 77-feature dataset                          │
│                    │                                         │
│                    ├──> File 03: Model Training             │
│                    │       ├──> LightGBM model (saved)      │
│                    │       ├──> Autoencoder model (saved)   │
│                    │       ├──> StandardScaler (saved)      │
│                    │       └──> Feature metadata (JSON)     │
│                    │                                         │
│                    └──> Validation metrics                  │
│                                                              │
│  MITRE ATT&CK Framework                                     │
│      │                                                       │
│      └──> File 04: RAG Explainer                            │
│            ├──> Knowledge base (47 techniques)              │
│            ├──> ChromaDB vector DB                          │
│            ├──> Embeddings (all-MiniLM-L6-v2)               │
│            └──> ProductionRAGExplainer class                │
│                                                              │
│  Zeek Network Monitor                                       │
│      │                                                       │
│      └──> File 05: Live Detection                           │
│            ├──> Zeek log ingestion                          │
│            ├──> Flow aggregation                            │
│            ├──> Feature engineering (same pipeline)         │
│            ├──> Load models from File 03                    │
│            ├──> 3-tier detection                            │
│            ├──> Load explainer from File 04                 │
│            ├──> Generate explanations                       │
│            └──> CSV export                                  │
│                                                              │
│  Llama3.1:8b LLM (Ollama)                           │
│      │                                                       │
│      └──> Called by RAG Explainer                           │
│            ├──> Generate natural language                   │
│            └──> Return explanation text                     │
│                                                              │
│  Output Systems                                             │
│      ├──> Console (formatted reports)                       │
│      ├──> CSV files (SIEM integration)                      │
│      └──> Logs (debugging/audit)                            │
└─────────────────────────────────────────────────────────────┘
**Critical Dependencies:**

1. **File 03 → File 05**: Trained models (LightGBM, Autoencoder, Scaler)
2. **File 04 → File 05**: RAG explainer class and MITRE knowledge base
3. **Zeek → File 05**: Real-time connection logs (conn.log)
4. **Ollama → File 04**: LLM runtime for explanation generation
5. **ChromaDB → File 04**: Vector database for semantic search

### 2.4 Network Topology

**Lab Network Architecture:**
```
Internet → Router (192.168.2.0/24)
    ↓
MAC-MINI Physical Host
├── pfSense A VM (Primary Firewall)
│   ├── WAN: 192.168.2.229/24 (DHCP from router)
│   └── LAN: 192.168.10.1/24 (Management Network)
│
└── Physical Kali Attack Box
    └── 192.168.10.101/24 (Bridged to physical network)
    
    ↓
    
Desktop Host (Windows) - VirtualBox Hypervisor
├── pfSense B VM (Internal Firewall)
│   ├── em0 (WAN): 192.168.10.5 - Bridged Adapter
│   ├── em1 (LAN): 192.168.30.1 - Internal Network
│   └── em2 (OPT1): 192.168.40.1 - UserLAN Network
│
├── Zeek VM (192.168.30.80) ← MONITORING HOST
│   ├── enp0s3: Internal Network (promiscuous mode)
│   └── enp0s8: UserLAN Network (promiscuous mode)
│   └── Cluster: Manager + Proxy + 2 Workers
│
├── Internal Network VMs (192.168.30.0/24)
│   ├── Wazuh SIEM (192.168.30.10)
│   ├── ELK Stack (192.168.30.20)
│   ├── Windows Server (192.168.30.40)
│   ├── MISP Threat Intel (192.168.30.50)
│   ├── OpenCTI (192.168.30.60)
│   ├── Shuffle SOAR (192.168.30.70)
│   └── DVWA WebServer (192.168.30.90)
│
└── User-LAN VMs (192.168.40.0/24)
    └── Metasploitable (192.168.40.10)

┌─────────────────────────────────────────────────────────┐
│ NIDS Monitoring Coverage                                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ ✓ Internal Network (192.168.30.0/24)                   │
│   - VM-to-VM traffic                                    │
│   - Attack scenarios from Internal VMs                  │
│   - Worker-1 monitoring via enp0s3                      │
│                                                         │
│ ✓ User-LAN (192.168.40.0/24)                           │
│   - Metasploitable vulnerable system                    │
│   - Attack traffic targeting User-LAN                   │
│   - Worker-2 monitoring via enp0s8                      │
│                                                         │
│ ✗ Management Network (192.168.10.0/24)                 │
│   - Physical Kali not monitored                         │
│   - Architecture constraints prevent coverage           │
└─────────────────────────────────────────────────────────┘
```

**Monitoring Scope:**
- **100% Coverage**: VM-to-VM attacks within Internal Network and User-LAN
- **0% Coverage**: Physical Kali → Virtual target attacks (routed traffic limitation)
- **Detection Rate**: 37.5% on benign traffic, 80-100% during active attacks

---

## 3. Scope and Objectives

### 3.1 Project Scope

**In Scope:**

**Data Processing:**
- CIC-IDS2017 dataset (2.3 million network flows)
- Friday-WorkingHours.pcap_ISCX.csv (331,170 samples)
- 14 attack types across 5 MITRE ATT&CK tactics
- 77 bidirectional flow features
- Train/test split with Friday hold-out validation

**Detection System:**
- Multi-tier ensemble architecture (3 tiers)
- LightGBM gradient boosting classifier
- Autoencoder-based anomaly detection
- Rule-based expert system
- Ensemble voting mechanism

**Explainability System:**
- RAG architecture with ChromaDB vector database
- 47 MITRE ATT&CK techniques
- Semantic search with embeddings
- Llama3.1:8b LLM integration
- Hallucination prevention mechanisms

**Integration:**
- Real-time Zeek log processing
- Flow aggregation and feature engineering
- CSV export for SIEM platforms
- Zeek UID correlation for forensics

**Out of Scope:**

**Not Implemented:**
- ❌ Real-time packet capture (uses Zeek logs)
- ❌ Deep packet inspection (DPI) beyond Zeek
- ❌ Inline blocking/prevention (IPS mode)
- ❌ Physical Kali → Virtual target monitoring (architecture limitation)
- ❌ Encrypted traffic analysis (SSL/TLS inspection)
- ❌ IPv6 attack detection (dataset limitation)
- ❌ Web-based dashboard/GUI
- ❌ Alerting system (email/Slack notifications)
- ❌ Automated incident response actions

**Future Enhancements (Deferred):**
- Advanced attack types (Web attacks, Infiltration)
- Multi-protocol analysis (HTTP, DNS, SSL deep dive)
- Threat intelligence feed integration (MISP)
- SIEM platform deployment (ELK/Wazuh)
- Machine learning model retraining pipeline
- A/B testing of detection algorithms

### 3.2 Primary Objectives

**Objective 1: High-Accuracy Attack Detection**
- **Target**: >95% accuracy
- **Achieved**: 99.89% test accuracy, 99.798% Friday hold-out
- **Metric**: ROC-AUC = 1.0000 (perfect discrimination)

**Objective 2: Multi-Tier Detection Architecture**
- **Target**: Combine 2-3 detection methods
- **Achieved**: 3-tier system (LightGBM + Autoencoder + Rules)
- **Benefit**: Ensemble consensus reduces false positives

**Objective 3: MITRE ATT&CK Integration**
- **Target**: Map attacks to ATT&CK framework
- **Achieved**: 47 techniques across 14 tactics
- **Method**: Semantic search + keyword matching

**Objective 4: Natural Language Explanations**
- **Target**: Generate human-readable threat reports
- **Achieved**: RAG + Llama3.1:8b with 66.7% LLM success rate
- **Quality**: No hallucinated techniques, complete sentences

**Objective 5: Real-Time Processing**
- **Target**: Process live Zeek logs
- **Achieved**: 200 connections → 8 flows in 1.87s
- **Throughput**: 4 flows/second

**Objective 6: Forensic Integration**
- **Target**: Enable incident investigation
- **Achieved**: Zeek UID correlation + timestamp preservation
- **Benefit**: Direct link to full packet capture

### 3.3 Learning Outcomes

**Machine Learning:**
- ✅ Imbalanced dataset handling (SMOTE, class weights)
- ✅ Gradient boosting hyperparameter tuning
- ✅ Autoencoder architecture design for anomaly detection
- ✅ Cross-validation strategies (StratifiedKFold)
- ✅ Model evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- ✅ Model persistence (joblib, Keras save)

**Deep Learning:**
- ✅ Neural network architecture design (encoder-decoder)
- ✅ Activation functions (ReLU, sigmoid)
- ✅ Loss functions (MSE for reconstruction)
- ✅ Optimizer selection (Adam)
- ✅ Early stopping and callbacks
- ✅ Feature scaling for neural networks

**Natural Language Processing:**
- ✅ Retrieval-Augmented Generation (RAG) architecture
- ✅ Vector databases (ChromaDB) and embeddings
- ✅ Semantic search with cosine similarity
- ✅ Large Language Model (LLM) integration via Ollama
- ✅ Prompt engineering for technical domains
- ✅ Hallucination prevention techniques

**Cybersecurity:**
- ✅ Network flow analysis (5-tuple, flags, IAT)
- ✅ Attack taxonomy (DoS, brute force, port scan, infiltration)
- ✅ MITRE ATT&CK framework navigation
- ✅ Threat intelligence report generation
- ✅ SIEM integration workflows
- ✅ Forensic correlation (Zeek UID)

**Systems Integration:**
- ✅ Zeek network monitoring deployment
- ✅ Real-time log parsing and aggregation
- ✅ Multi-process pipeline orchestration
- ✅ Error handling and fallback mechanisms
- ✅ CSV export for external systems
- ✅ Production-ready code organization

**Problem Solving:**
- ✅ Debugging truncated LLM outputs
- ✅ Removing LLM "thinking process" artifacts
- ✅ Resolving sklearn feature name warnings
- ✅ Adapting to infrastructure constraints (VM networking)
- ✅ Balancing detection accuracy vs false positive rate

### 3.4 Excluded Scope

**Consciously Excluded:**

1. **Physical Network Monitoring**
   - **Reason**: VirtualBox architecture constraints
   - **Alternative**: Snort IDS inline at routing boundary (future project)
   - **Impact**: 0% coverage for physical Kali → virtual target attacks

2. **Real-Time Alerting**
   - **Reason**: Focus on detection and explanation first
   - **Future**: Integration with Slack, email, or PagerDuty
   - **Workaround**: CSV export enables SIEM alerting

3. **Web Dashboard**
   - **Reason**: Time constraints and scope management
   - **Future**: Kibana/Grafana visualization
   - **Workaround**: Console output and CSV reports

4. **Automated Response**
   - **Reason**: Safety concerns in lab environment
   - **Future**: Integration with Shuffle SOAR for playbooks
   - **Workaround**: Manual investigation using Zeek UID

5. **Advanced Attack Types**
   - **Excluded**: Web attacks (XSS, SQLi), Infiltration
   - **Reason**: Insufficient training samples (<1000 per class)
   - **Future**: Additional datasets or synthetic data generation

---

## 4. Prerequisites

### 4.1 Infrastructure Requirements

**Development Machine:**
- **OS**: Windows 10/11, Linux, or macOS
- **CPU**: 4 cores minimum (8 cores recommended for training)
- **RAM**: 16 GB minimum (32 GB recommended)
- **Storage**: 100 GB free space
  - Dataset: 1.5 GB (CSV files)
  - Models: 500 MB (LightGBM + Autoencoder)
  - Zeek logs: 10-20 GB (with rotation)
  - ChromaDB: 50 MB (vector database)
- **GPU**: Optional (TensorFlow can use CUDA for autoencoder)

**Zeek Monitoring VM:**
- **OS**: Ubuntu 22.04 LTS
- **CPU**: 2 cores minimum
- **RAM**: 4 GB minimum
- **Storage**: 50 GB (for log rotation)
- **Network**: 2 network interfaces (promiscuous mode)
  - enp0s3: Internal Network (192.168.30.0/24)
  - enp0s8: UserLAN Network (192.168.40.0/24)

**Network Architecture:**
- VirtualBox hypervisor with internal networks
- pfSense firewall for routing
- Vulnerable VMs for attack simulation (Metasploitable, DVWA)
- Security tools (Wazuh, ELK, MISP) for context

### 4.2 Software Dependencies

**Python Environment:**
```bash
# Core data science
pandas==2.2.0
numpy==1.26.0
matplotlib==3.8.0
seaborn==0.13.0

# Machine learning
scikit-learn==1.5.0
lightgbm==4.5.0
imbalanced-learn==0.12.0
joblib==1.3.0

# Deep learning
tensorflow==2.17.0
keras==3.4.0

# NLP and embeddings
sentence-transformers==2.7.0
chromadb==0.4.24

# Utilities
tqdm==4.66.0
python-dateutil==2.9.0
```

**External Services:**
```bash
# LLM runtime
ollama (with llama3.1:8b model)

# Network monitoring
zeek 6.0.3 (compiled from source)

# Vector database
chromadb-server (embedded mode)
```

**Installation Commands:**
```bash
# Create virtual environment
python -m venv nids-venv
source nids-venv/bin/activate  # Linux/Mac
nids-venv\Scripts\activate  # Windows

# Install Python packages
pip install pandas numpy matplotlib seaborn
pip install scikit-learn lightgbm imbalanced-learn joblib
pip install tensorflow keras
pip install sentence-transformers chromadb
pip install tqdm python-dateutil

# Install Ollama (Linux)
curl https://ollama.ai/install.sh | sh
ollama pull llama3.1:8b

# Zeek installation (Ubuntu)
# See File 05 documentation for compilation from source
```

### 4.3 Dataset Requirements

**CIC-IDS2017 Dataset:**
- **Source**: Canadian Institute for Cybersecurity
- **URL**: https://www.unb.ca/cic/datasets/ids-2017.html
- **Size**: 8 CSV files, ~2.3 GB total
- **Format**: CSV with 79 features + 1 label column

**Required Files:**
```
CIC-IDS2017/
├── Monday-WorkingHours.pcap_ISCX.csv (529,918 samples)
├── Tuesday-WorkingHours.pcap_ISCX.csv (445,909 samples)
├── Wednesday-WorkingHours.pcap_ISCX.csv (692,703 samples)
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv (170,366 samples)
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv (288,602 samples)
├── Friday-WorkingHours-Morning.pcap_ISCX.csv (191,033 samples)
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv (286,467 samples)
└── Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv (225,745 samples)
```

**Primary Analysis File:**
- `Friday-WorkingHours.pcap_ISCX.csv` (331,170 samples)
- Contains diverse attack types for comprehensive testing
- Represents realistic mix of benign and malicious traffic

**Attack Types in Dataset:**
```
Benign:               97,718 samples (29.5%)
DoS Hulk:             46,240 samples (14.0%)
DDoS:                 41,835 samples (12.6%)
PortScan:             31,649 samples (9.6%)
DoS GoldenEye:        10,293 samples (3.1%)
FTP-Patator:           7,938 samples (2.4%)
SSH-Patator:           5,897 samples (1.8%)
DoS slowloris:         5,796 samples (1.8%)
DoS Slowhttptest:      5,499 samples (1.7%)
Bot:                   1,966 samples (0.6%)
Web Attack - Brute Force:   1,507 samples (0.5%)
Web Attack - XSS:        652 samples (0.2%)
Infiltration:            36 samples (0.01%)
Web Attack - Sql Injection:  21 samples (0.01%)
Heartbleed:              11 samples (0.003%)
```

### 4.4 Knowledge Prerequisites

**Required Background:**

**Python Programming:**
- ✅ Pandas dataframe operations
- ✅ NumPy array manipulation
- ✅ Function and class definitions
- ✅ File I/O and path handling
- ✅ Exception handling

**Machine Learning Fundamentals:**
- ✅ Supervised vs unsupervised learning
- ✅ Classification vs regression
- ✅ Train/test split and cross-validation
- ✅ Overfitting and regularization
- ✅ Model evaluation metrics

**Cybersecurity Basics:**
- ✅ Network protocols (TCP, UDP, ICMP)
- ✅ Common attack types (DoS, brute force, port scan)
- ✅ Network flow analysis (5-tuple)
- ✅ MITRE ATT&CK framework basics

**Recommended (Not Required):**
- Neural networks and deep learning concepts
- Natural language processing basics
- Vector databases and embeddings
- Zeek/Bro network monitoring
- SIEM platform concepts

**Learning Resources:**
- Scikit-learn documentation: https://scikit-learn.org/
- LightGBM documentation: https://lightgbm.readthedocs.io/
- MITRE ATT&CK: https://attack.mitre.org/
- Zeek documentation: https://docs.zeek.org/
- RAG concepts: Anthropic/OpenAI research papers

---

## 5. Implementation: File 01 - Dataset Preparation

### 5.1 Dataset Overview (CIC-IDS2017)

**Project Background:**

The CIC-IDS2017 dataset was created by the Canadian Institute for Cybersecurity to address limitations in existing intrusion detection datasets. Unlike older datasets (KDD Cup 99, NSL-KDD), CIC-IDS2017 includes:
- Modern attack scenarios (2017 threat landscape)
- Realistic network traffic profiles
- Labeled PCAP files with extracted flow features
- Diverse attack types across multiple days

**Dataset Statistics:**
```
┌─────────────────────────────────────────────────────────────┐
│              CIC-IDS2017 DATASET OVERVIEW                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Total Samples:     2,830,743 network flows                │
│  Total Size:        ~2.3 GB (8 CSV files)                  │
│  Features:          79 features + 1 label                   │
│  Attack Types:      14 distinct attacks                     │
│  Days Captured:     5 days (Monday - Friday)               │
│  Network:           Simulated enterprise environment        │
│                                                             │
│  Benign Traffic:    2,273,097 samples (80.3%)              │
│  Attack Traffic:    557,646 samples (19.7%)                │
│                                                             │
│  Class Imbalance:   Severe (some attacks <100 samples)     │
│  Data Quality:      Some inf/nan values requiring cleanup   │
└─────────────────────────────────────────────────────────────┘
```

**Why Friday-WorkingHours.pcap_ISCX.csv?**

For this implementation, we focused on the Friday capture file because:
1. **Comprehensive Attack Coverage**: Contains 14 different attack types
2. **Realistic Mix**: 29.5% benign, 70.5% attacks (challenging scenario)
3. **Manageable Size**: 331,170 samples (fits in memory)
4. **Diverse Tactics**: Covers Impact, Credential Access, Discovery, C2, and Initial Access
5. **Quality**: Clean labels with minimal preprocessing required

### 5.2 Exploratory Data Analysis

**Step 1: Initial Data Loading**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('Friday-WorkingHours.pcap_ISCX.csv')

# Basic information
print(f"Dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes.value_counts()}")
```

**Output:**
```
Dataset shape: (331170, 79)
Memory usage: 199.45 MB

Data types:
float64    77
object      2
```

**Key Observations:**
- 77 numerical features (flow statistics)
- 2 object columns (Label, Source/Destination IP in some versions)
- All features are float64 (no integers)

**Step 2: Missing Value Analysis**
```python
# Check for missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
}).sort_values('Percentage', ascending=False)

print(missing_df[missing_df['Percentage'] > 0].head(10))
```

**Output:**
```
                              Missing Count  Percentage
Flow Bytes/s                          1918    0.579%
Flow Packets/s                        1918    0.579%
```

**Step 3: Infinite Value Detection**
```python
# Check for infinite values
inf_counts = {}
for col in df.select_dtypes(include=[np.number]).columns:
    inf_count = np.isinf(df[col]).sum()
    if inf_count > 0:
        inf_counts[col] = inf_count

print(f"Columns with infinite values: {len(inf_counts)}")
for col, count in sorted(inf_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
```

**Output:**
```
Columns with infinite values: 3
  Flow Bytes/s: 1918 (0.58%)
  Flow Packets/s: 1918 (0.58%)
  Bwd Avg Bulk Rate: 3 (0.001%)
```

**Step 4: Class Distribution**
```python
# Label distribution
label_counts = df[' Label'].value_counts()
label_pct = (label_counts / len(df)) * 100

print("Attack Type Distribution:")
for label, count in label_counts.items():
    print(f"  {label:30s}: {count:7d} ({label_pct[label]:5.2f}%)")
```

**Output:**
```
Attack Type Distribution:
  BENIGN                        :   97718 (29.50%)
  DoS Hulk                      :   46240 (13.96%)
  DDoS                          :   41835 (12.63%)
  PortScan                      :   31649 ( 9.56%)
  DoS GoldenEye                 :   10293 ( 3.11%)
  FTP-Patator                   :    7938 ( 2.40%)
  SSH-Patator                   :    5897 ( 1.78%)
  DoS slowloris                 :    5796 ( 1.75%)
  DoS Slowhttptest              :    5499 ( 1.66%)
  Bot                           :    1966 ( 0.59%)
  Web Attack   Brute Force      :    1507 ( 0.45%)
  Web Attack   XSS              :     652 ( 0.20%)
  Infiltration                  :      36 ( 0.01%)
  Web Attack   Sql Injection    :      21 ( 0.01%)
  Heartbleed                    :      11 ( 0.003%)
```

**Key Insight**: Severe class imbalance, with Heartbleed having only 11 samples!

### 5.3 Data Quality Assessment

**Issue 1: Infinite Values**

**Problem**: Division by zero in rate calculations (Flow Bytes/s, Flow Packets/s)

**Root Cause**: Flows with duration = 0

**Solution**:
```python
# Replace infinite values with 0 (flow lasted 0 seconds)
df.replace([np.inf, -np.inf], 0, inplace=True)
```

**Issue 2: Missing Values**

**Problem**: 1,918 rows with NaN in Flow Bytes/s and Flow Packets/s

**Solution**:
```python
# Fill missing rate values with 0
df[' Flow Bytes/s'].fillna(0, inplace=True)
df[' Flow Packets/s'].fillna(0, inplace=True)
```

**Issue 3: Column Name Inconsistencies**

**Problem**: Leading/trailing spaces in column names (' Label', ' Flow Bytes/s')

**Solution**:
```python
# Strip whitespace from column names
df.columns = df.columns.str.strip()
```

**Issue 4: Class Imbalance**

**Problem**: 
- Benign: 97,718 samples
- Heartbleed: 11 samples
- Ratio: 8,883:1 imbalance!

**Solution**: SMOTE (Synthetic Minority Over-sampling Technique) in File 02

**Issue 5: Feature Scale Differences**

**Problem**: Features range from 0 to millions (e.g., Total Fwd Packets vs Flow Bytes/s)

**Solution**: StandardScaler normalization before modeling

### 5.4 Class Distribution Analysis

**Visualization:**
```python
# Plot class distribution
plt.figure(figsize=(14, 6))
label_counts.plot(kind='barh', color='steelblue')
plt.xlabel('Number of Samples')
plt.ylabel('Attack Type')
plt.title('CIC-IDS2017 Friday: Attack Type Distribution')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300)
plt.show()
```

**Attack Taxonomy Grouping:**
```
┌─────────────────────────────────────────────────────────────┐
│           ATTACK TAXONOMY BY MITRE TACTIC                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  IMPACT (Denial of Service):                               │
│    ├─ DoS Hulk:          46,240 samples                    │
│    ├─ DDoS:              41,835 samples                    │
│    ├─ DoS GoldenEye:     10,293 samples                    │
│    ├─ DoS slowloris:      5,796 samples                    │
│    └─ DoS Slowhttptest:   5,499 samples                    │
│       Total:             109,663 samples (33.1%)           │
│                                                             │
│  DISCOVERY (Network Reconnaissance):                        │
│    └─ PortScan:          31,649 samples (9.6%)             │
│                                                             │
│  CREDENTIAL ACCESS (Brute Force):                           │
│    ├─ FTP-Patator:        7,938 samples                    │
│    └─ SSH-Patator:        5,897 samples                    │
│       Total:             13,835 samples (4.2%)             │
│                                                             │
│  COMMAND & CONTROL:                                         │
│    └─ Bot:                1,966 samples (0.6%)             │
│                                                             │
│  INITIAL ACCESS (Web Exploits):                             │
│    ├─ Web Attack - Brute Force:  1,507 samples             │
│    ├─ Web Attack - XSS:            652 samples             │
│    ├─ Web Attack - SQL Injection:   21 samples             │
│    ├─ Infiltration:                 36 samples             │
│    └─ Heartbleed:                   11 samples             │
│       Total:              2,227 samples (0.7%)             │
└─────────────────────────────────────────────────────────────┘
```

**Training Strategy Implications:**

1. **Exclude Rare Classes**: Heartbleed (11), SQLi (21), Infiltration (36) dropped
2. **Group Similar Attacks**: All DoS variants → "DoS" category
3. **SMOTE for Minority**: Oversample Bot, Web attacks, Patator attacks
4. **Stratified Split**: Maintain class ratios in train/test

### 5.5 Feature Space Exploration

**Feature Categories:**

The 77 features in CIC-IDS2017 can be grouped into 9 categories:

**1. Basic Flow Information (5 features)**
```
- Flow Duration: Total time of the flow (microseconds)
- Flow Bytes/s: Rate of bytes per second
- Flow Packets/s: Rate of packets per second
- Total Fwd Packets: Number of forward packets
- Total Backward Packets: Number of backward packets
```

**2. Packet Length Statistics (14 features)**
```
Forward direction:
- Total Length of Fwd Packets
- Fwd Packet Length Max/Min/Mean/Std

Backward direction:
- Total Length of Bwd Packets
- Bwd Packet Length Max/Min/Mean/Std

Overall:
- Packet Length Max/Min/Mean/Std/Variance
```

**3. Inter-Arrival Time (IAT) Statistics (8 features)**
```
- Flow IAT Mean/Std/Max/Min
- Fwd IAT Total/Mean/Std/Max/Min
- Bwd IAT Total/Mean/Std/Max/Min
```

**4. TCP Flags (6 features)**
```
- FIN Flag Count
- SYN Flag Count
- RST Flag Count
- PSH Flag Count
- ACK Flag Count
- URG Flag Count
```

**5. Header Information (4 features)**
```
- Fwd/Bwd Header Length (total)
- Fwd/Bwd Avg Header Bytes
```

**6. Bulk Transfer Rates (3 features)**
```
- Fwd Avg Bytes/Bulk
- Fwd Avg Packets/Bulk
- Fwd Avg Bulk Rate
- Bwd Avg Bytes/Bulk
- Bwd Avg Packets/Bulk
- Bwd Avg Bulk Rate
```

**7. Subflow Statistics (3 features)**
```
- Subflow Fwd Packets
- Subflow Fwd Bytes
- Subflow Bwd Packets
- Subflow Bwd Bytes
```

**8. Active/Idle Time (4 features)**
```
- Active Mean/Std/Max/Min
- Idle Mean/Std/Max/Min
```

**9. Segment Size (2 features)**
```
- Fwd Avg Segment Size
- Bwd Avg Segment Size
```

**Feature Correlation Analysis:**
```python
# Calculate correlation matrix
corr_matrix = df.select_dtypes(include=[np.number]).corr()

# Find highly correlated features (>0.95)
high_corr = []
for i in range(len(corr_matrix.columns)):
for j in range(i+1, len(corr_matrix.columns)):
if abs(corr_matrix.iloc[i, j]) > 0.95:
high_corr.append((
corr_matrix.columns[i],
corr_matrix.columns[j],
corr_matrix.iloc[i, j]
))
print(f"Found {len(high_corr)} highly correlated feature pairs:")
for feat1, feat2, corr in high_corr[:10]:
print(f"  {feat1:40s} <-> {feat2:40s}: {corr:.3f}")
**Output:**
```
Found 23 highly correlated feature pairs:
  Total Length of Fwd Packets           <-> Fwd Packet Length Max            : 0.982
  Total Fwd Packets                     <-> Subflow Fwd Packets              : 0.999
  Total Backward Packets                <-> Subflow Bwd Packets              : 0.999
  Flow Bytes/s                          <-> Flow Packets/s                   : 0.967
  Fwd Header Length                     <-> Total Fwd Packets                : 0.978
```

**Key Findings:**
- Many redundant features (e.g., Total vs Subflow)
- Packet length features highly correlated
- Rate features (bytes/s, packets/s) correlated
- Opportunity for dimensionality reduction (but kept all 77 for interpretability)

**Feature Distribution Analysis:**
```python
# Analyze feature distributions by class
benign = df[df['Label'] == 'BENIGN']
attack = df[df['Label'] != 'BENIGN']

# Key discriminative features
key_features = [
    'Flow Duration',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Total Fwd Packets',
    'Total Backward Packets',
    'SYN Flag Count',
    'RST Flag Count',
    'ACK Flag Count'
]

fig, axes = plt.subplots(4, 2, figsize=(14, 16))
axes = axes.flatten()

for i, feature in enumerate(key_features):
    benign[feature].hist(bins=50, alpha=0.5, label='Benign', ax=axes[i], color='green')
    attack[feature].hist(bins=50, alpha=0.5, label='Attack', ax=axes[i], color='red')
    axes[i].set_title(feature)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    axes[i].legend()
    axes[i].set_yscale('log')  # Log scale for better visualization

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300)
plt.show()
```

**Observations:**

1. **Flow Duration**: Attacks tend to have shorter durations (automated)
2. **Flow Bytes/s**: Attacks show higher rates (especially DoS)
3. **SYN Flag Count**: Port scans have high SYN counts with few ACKs
4. **RST Flag Count**: Brute force attacks trigger many RST (rejected connections)
5. **Packet Counts**: Attack flows vary widely (DoS: high, port scan: low)

**Feature Importance Preview (from LightGBM):**
```
Top 10 Most Important Features:
1.  Flow Bytes/s                 : 0.142 (14.2%)
2.  Total Fwd Packets            : 0.089 (8.9%)
3.  Flow Duration                : 0.076 (7.6%)
4.  Fwd Packet Length Mean       : 0.068 (6.8%)
5.  Flow Packets/s               : 0.062 (6.2%)
6.  Total Length of Fwd Packets  : 0.059 (5.9%)
7.  Bwd Packet Length Mean       : 0.054 (5.4%)
8.  SYN Flag Count               : 0.047 (4.7%)
9.  Fwd IAT Mean                 : 0.043 (4.3%)
10. Total Backward Packets       : 0.041 (4.1%)
```

**Data Preprocessing Summary:**
```python
# Final preprocessing steps
print("="*70)
print("DATA PREPROCESSING SUMMARY")
print("="*70)

# 1. Handle infinite values
df.replace([np.inf, -np.inf], 0, inplace=True)
print("✓ Replaced infinite values with 0")

# 2. Handle missing values
df.fillna(0, inplace=True)
print("✓ Filled missing values with 0")

# 3. Strip column names
df.columns = df.columns.str.strip()
print("✓ Cleaned column names")

# 4. Remove rare classes
min_samples = 100
label_counts = df['Label'].value_counts()
rare_labels = label_counts[label_counts < min_samples].index.tolist()
df = df[~df['Label'].isin(rare_labels)]
print(f"✓ Removed {len(rare_labels)} rare classes: {rare_labels}")

# 5. Save cleaned dataset
df.to_parquet('Friday-WorkingHours-clean.parquet', index=False)
print("✓ Saved cleaned dataset to parquet format")

print(f"\nFinal dataset shape: {df.shape}")
print(f"Final classes: {df['Label'].nunique()}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

**Output:**
```
======================================================================
DATA PREPROCESSING SUMMARY
======================================================================
✓ Replaced infinite values with 0
✓ Filled missing values with 0
✓ Cleaned column names
✓ Removed 3 rare classes: ['Heartbleed', 'Web Attack   Sql Injection', 'Infiltration']
✓ Saved cleaned dataset to parquet format

Final dataset shape: (331102, 79)
Final classes: 12
Memory usage: 198.87 MB
```

**File 01 Deliverables:**
```
outputs/
├── Friday-WorkingHours-clean.parquet     (331,102 samples, 12 classes)
├── class_distribution.png                (visualization)
├── feature_distributions.png             (visualization)
└── eda_summary.txt                       (statistics)
```

**Key Takeaways from File 01:**

✅ **Data Quality**: Clean dataset with inf/NaN handled  
✅ **Class Balance**: Identified severe imbalance requiring SMOTE  
✅ **Feature Understanding**: 77 features grouped into 9 categories  
✅ **Attack Taxonomy**: Mapped to MITRE ATT&CK tactics  
✅ **Rare Classes**: Removed 3 classes with <100 samples  
✅ **Ready for Feature Engineering**: Preprocessed data saved in efficient format  

---

## 6. Implementation: File 02 - Feature Engineering

### 6.1 Feature Engineering Strategy

**Objective**: Transform raw CIC-IDS2017 features into a robust feature space optimized for machine learning while maintaining interpretability for security analysts.

**Challenges:**
1. High dimensionality (79 raw features)
2. Severe class imbalance (8,883:1 ratio)
3. Feature redundancy (correlation >0.95)
4. Scale differences (0 to millions)
5. Attack-specific feature importance

**Strategy:**
```
┌─────────────────────────────────────────────────────────────┐
│           FEATURE ENGINEERING PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  STAGE 1: Feature Selection                                │
│    ├─ Keep all 77 numeric features (interpretability)      │
│    ├─ Remove redundant label columns                        │
│    └─ Preserve domain-specific features (flags, IAT)        │
│                                                             │
│  STAGE 2: Feature Transformation                            │
│    ├─ Log transformation for heavy-tailed distributions     │
│    ├─ Polynomial features (selected pairs only)             │
│    └─ Bidirectional flow features (Fwd + Bwd)              │
│                                                             │
│  STAGE 3: Feature Scaling                                  │
│    ├─ StandardScaler (zero mean, unit variance)            │
│    ├─ Fit on training set only (prevent leakage)           │
│    └─ Transform train/test/production data                  │
│                                                             │
│  STAGE 4: Class Balancing                                  │
│    ├─ SMOTE oversampling for minority classes              │
│    ├─ Random undersampling for majority class              │
│    └─ Target: Balanced dataset for training                 │
│                                                             │
│  STAGE 5: Validation                                       │
│    ├─ Feature distribution analysis                         │
│    ├─ Correlation matrix validation                         │
│    └─ Train/test split with stratification                  │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Bidirectional Flow Features

**Concept**: Network flows have directionality - forward (client → server) and backward (server → client). Combining these captures the full conversation.

**Implementation:**
```python
import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_parquet('Friday-WorkingHours-clean.parquet')

# Separate features and labels
X = df.drop('Label', axis=1)
y = df['Label']

print(f"Original feature space: {X.shape}")

# Bidirectional feature engineering
bidirectional_features = []

# 1. Packet Count Features
X['Total_Packets'] = X['Total Fwd Packets'] + X['Total Backward Packets']
X['Packet_Asymmetry'] = (X['Total Fwd Packets'] - X['Total Backward Packets']) / \
                         (X['Total Fwd Packets'] + X['Total Backward Packets'] + 1e-6)
bidirectional_features.extend(['Total_Packets', 'Packet_Asymmetry'])

# 2. Byte Count Features
X['Total_Bytes'] = X['Total Length of Fwd Packets'] + X['Total Length of Bwd Packets']
X['Byte_Asymmetry'] = (X['Total Length of Fwd Packets'] - X['Total Length of Bwd Packets']) / \
                       (X['Total Length of Fwd Packets'] + X['Total Length of Bwd Packets'] + 1e-6)
bidirectional_features.extend(['Total_Bytes', 'Byte_Asymmetry'])

# 3. Packet Length Features
X['Avg_Packet_Length'] = (X['Fwd Packet Length Mean'] + X['Bwd Packet Length Mean']) / 2
X['Packet_Length_Variance'] = ((X['Fwd Packet Length Std'] ** 2) + 
                                (X['Bwd Packet Length Std'] ** 2)) / 2
bidirectional_features.extend(['Avg_Packet_Length', 'Packet_Length_Variance'])

# 4. IAT Features
X['Avg_IAT'] = (X['Fwd IAT Mean'] + X['Bwd IAT Mean']) / 2
X['IAT_Variance'] = ((X['Fwd IAT Std'] ** 2) + (X['Bwd IAT Std'] ** 2)) / 2
bidirectional_features.extend(['Avg_IAT', 'IAT_Variance'])

# 5. Header Features
X['Total_Header_Length'] = X['Fwd Header Length'] + X['Bwd Header Length']
X['Header_Asymmetry'] = (X['Fwd Header Length'] - X['Bwd Header Length']) / \
                         (X['Fwd Header Length'] + X['Bwd Header Length'] + 1e-6)
bidirectional_features.extend(['Total_Header_Length', 'Header_Asymmetry'])

print(f"Added {len(bidirectional_features)} bidirectional features")
print(f"New feature space: {X.shape}")
```

**Output:**
```
Original feature space: (331102, 77)
Added 10 bidirectional features
New feature space: (331102, 87)
```

**Rationale for Each Feature:**

| Feature | Rationale | Attack Detection Value |
|---------|-----------|------------------------|
| **Total_Packets** | Overall flow size | DoS: Very high, Port scan: Very low |
| **Packet_Asymmetry** | Directionality ratio | Port scan: High asymmetry (only SYN), Normal: Balanced |
| **Total_Bytes** | Data volume | DoS: High volume, Brute force: Low volume |
| **Byte_Asymmetry** | Data flow direction | Upload/download patterns, C2: Balanced bidirectional |
| **Avg_Packet_Length** | Average payload size | Web attacks: Larger payloads, DoS: Smaller packets |
| **Packet_Length_Variance** | Payload consistency | Normal: Variable, Automated: Consistent |
| **Avg_IAT** | Time between packets | Automated: Consistent timing, Human: Variable |
| **IAT_Variance** | Timing consistency | Botnet: Low variance, Normal: High variance |
| **Total_Header_Length** | Protocol overhead | Port scan: Header-only packets |
| **Header_Asymmetry** | Protocol imbalance | SYN flood: High asymmetry (no SYN-ACK) |

### 6.3 Statistical Aggregations

**Concept**: Create summary statistics that capture overall flow behavior.

**Implementation:**
```python
# 6. Flag-based Features
X['Total_Flags'] = (X['FIN Flag Count'] + X['SYN Flag Count'] + 
                    X['RST Flag Count'] + X['PSH Flag Count'] + 
                    X['ACK Flag Count'] + X['URG Flag Count'])

X['Flag_Diversity'] = 0
for flag in ['FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 
             'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count']:
    X['Flag_Diversity'] += (X[flag] > 0).astype(int)

X['SYN_ACK_Ratio'] = X['SYN Flag Count'] / (X['ACK Flag Count'] + 1e-6)
X['RST_PSH_Ratio'] = X['RST Flag Count'] / (X['PSH Flag Count'] + 1e-6)

bidirectional_features.extend(['Total_Flags', 'Flag_Diversity', 
                               'SYN_ACK_Ratio', 'RST_PSH_Ratio'])

# 7. Rate-based Features
X['Bytes_Per_Packet'] = X['Total_Bytes'] / (X['Total_Packets'] + 1e-6)
X['Packets_Per_Second'] = X['Total_Packets'] / (X['Flow Duration'] / 1_000_000 + 1e-6)
X['Bytes_Per_Second'] = X['Total_Bytes'] / (X['Flow Duration'] / 1_000_000 + 1e-6)

bidirectional_features.extend(['Bytes_Per_Packet', 'Packets_Per_Second', 
                               'Bytes_Per_Second'])

# 8. Temporal Features
X['Flow_Duration_Sec'] = X['Flow Duration'] / 1_000_000  # Convert to seconds
X['Active_Idle_Ratio'] = X['Active Mean'] / (X['Idle Mean'] + 1e-6)
X['Active_Percentage'] = X['Active Mean'] / (X['Active Mean'] + X['Idle Mean'] + 1e-6)

bidirectional_features.extend(['Flow_Duration_Sec', 'Active_Idle_Ratio', 
                               'Active_Percentage'])

print(f"Final feature space: {X.shape}")
print(f"Total engineered features: {len(bidirectional_features)}")
```

**Output:**
```
Final feature space: (331102, 100)
Total engineered features: 23
```

**Feature Engineering Results:**
- Started with: 77 features
- Added: 23 engineered features
- Final: 100 features (but we use 77 for consistency with training)

### 6.4 Feature Selection and Validation

**Approach**: Keep all 77 original features for interpretability, validate quality.
```python
# Feature validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Use only the 77 original features (bidirectional features optional)
feature_cols = [col for col in X.columns if col not in bidirectional_features]
X_selected = X[feature_cols]

print(f"Selected features: {len(feature_cols)}")

# Check for remaining issues
print("\n" + "="*70)
print("FEATURE QUALITY VALIDATION")
print("="*70)

# 1. Infinite values
inf_count = np.isinf(X_selected).sum().sum()
print(f"Infinite values: {inf_count}")

# 2. Missing values
nan_count = X_selected.isnull().sum().sum()
print(f"Missing values: {nan_count}")

# 3. Zero variance features
zero_var = (X_selected.std() == 0).sum()
print(f"Zero variance features: {zero_var}")

# 4. Feature range
print(f"\nFeature ranges:")
print(f"  Min: {X_selected.min().min():.2f}")
print(f"  Max: {X_selected.max().max():.2e}")
print(f"  Mean: {X_selected.mean().mean():.2f}")
print(f"  Std: {X_selected.std().mean():.2f}")

# 5. Correlation analysis
corr_matrix = X_selected.corr()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.95:
            high_corr_pairs.append((corr_matrix.columns[i], 
                                    corr_matrix.columns[j], 
                                    corr_matrix.iloc[i, j]))

print(f"\nHighly correlated pairs (>0.95): {len(high_corr_pairs)}")
```

**Output:**
```
Selected features: 77

======================================================================
FEATURE QUALITY VALIDATION
======================================================================
Infinite values: 0
Missing values: 0
Zero variance features: 0

Feature ranges:
  Min: 0.00
  Max: 1.26e+09
  Mean: 45237.88
  Std: 12438.92

Highly correlated pairs (>0.95): 23
```

### 6.5 Data Preprocessing Pipeline

**Complete Pipeline:**
```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

print("="*70)
print("DATA PREPROCESSING PIPELINE")
print("="*70)

# STEP 1: Encode labels
print("\n[1/6] Encoding labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"✓ Encoded {len(le.classes_)} classes")
print(f"  Classes: {le.classes_.tolist()}")

# Save label encoder
joblib.dump(le, 'models/label_encoder.joblib')
print("✓ Saved label encoder")

# STEP 2: Train/test split
print("\n[2/6] Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)
print(f"✓ Train: {X_train.shape[0]} samples")
print(f"✓ Test: {X_test.shape[0]} samples")

# STEP 3: Handle class imbalance
print("\n[3/6] Handling class imbalance with SMOTE...")
print("Before SMOTE:")
unique, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  {le.classes_[label]:30s}: {count:6d}")

# Define resampling strategy
smote = SMOTE(random_state=42, k_neighbors=5)
undersample = RandomUnderSampler(random_state=42)

# Apply SMOTE
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE:")
unique, counts = np.unique(y_train_balanced, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  {le.classes_[label]:30s}: {count:6d}")

print(f"\n✓ Balanced training set: {X_train_balanced.shape[0]} samples")

# STEP 4: Feature scaling
print("\n[4/6] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

print("✓ StandardScaler fitted on training data")
print(f"  Mean: {scaler.mean_[:5]} ...")
print(f"  Std: {scaler.scale_[:5]} ...")

# Save scaler
joblib.dump(scaler, 'models/feature_scaler.joblib')
print("✓ Saved feature scaler")

# STEP 5: Save processed datasets
print("\n[5/6] Saving processed datasets...")
np.save('data/X_train_scaled.npy', X_train_scaled)
np.save('data/X_test_scaled.npy', X_test_scaled)
np.save('data/y_train.npy', y_train_balanced)
np.save('data/y_test.npy', y_test)

# Also save feature names
feature_metadata = {
    'feature_names': feature_cols,
    'n_features': len(feature_cols),
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist()
}
import json
with open('data/feature_metadata.json', 'w') as f:
    json.dump(feature_metadata, f, indent=2)

print("✓ Saved training/test sets")
print("✓ Saved feature metadata")

# STEP 6: Validation summary
print("\n[6/6] Validation summary...")
print(f"\nTraining Set:")
print(f"  Shape: {X_train_scaled.shape}")
print(f"  Mean: {X_train_scaled.mean():.6f}")
print(f"  Std: {X_train_scaled.std():.6f}")
print(f"  Min: {X_train_scaled.min():.6f}")
print(f"  Max: {X_train_scaled.max():.6f}")

print(f"\nTest Set:")
print(f"  Shape: {X_test_scaled.shape}")
print(f"  Mean: {X_test_scaled.mean():.6f}")
print(f"  Std: {X_test_scaled.std():.6f}")
print(f"  Min: {X_test_scaled.min():.6f}")
print(f"  Max: {X_test_scaled.max():.6f}")

print("\n" + "="*70)
print("✅ DATA PREPROCESSING COMPLETE")
print("="*70)
```

**Output:**
```
======================================================================
DATA PREPROCESSING PIPELINE
======================================================================

[1/6] Encoding labels...
✓ Encoded 12 classes
  Classes: ['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 
             'DoS Slowhttptest', 'DoS slowloris', 'FTP-Patator', 
             'PortScan', 'SSH-Patator', 'Web Attack   Brute Force', 
             'Web Attack   XSS']
✓ Saved label encoder

[2/6] Splitting train/test...
✓ Train: 264881 samples
✓ Test: 66221 samples

[3/6] Handling class imbalance with SMOTE...
Before SMOTE:
  BENIGN                        :  78174
  Bot                           :   1572
  DDoS                          :  33468
  DoS GoldenEye                 :   8234
  DoS Hulk                      :  36992
  DoS Slowhttptest              :   4399
  DoS slowloris                 :   4636
  FTP-Patator                   :   6350
  PortScan                      :  25319
  SSH-Patator                   :   4717
  Web Attack   Brute Force      :   1205
  Web Attack   XSS              :    521

After SMOTE:
  BENIGN                        :  78174
  Bot                           :  78174
  DDoS                          :  78174
  DoS GoldenEye                 :  78174
  DoS Hulk                      :  78174
  DoS Slowhttptest              :  78174
  DoS slowloris                 :  78174
  FTP-Patator                   :  78174
  PortScan                      :  78174
  SSH-Patator                   :  78174
  Web Attack   Brute Force      :  78174
  Web Attack   XSS              :  78174

✓ Balanced training set: 938088 samples

[4/6] Scaling features...
✓ StandardScaler fitted on training data
  Mean: [ 3.14e+04  1.68e+05  1.21e+02  7.93e+01  4.31e+01] ...
  Std: [ 2.56e+05  8.92e+05  4.28e+02  3.21e+02  1.87e+02] ...
✓ Saved feature scaler

[5/6] Saving processed datasets...
✓ Saved training/test sets
✓ Saved feature metadata

[6/6] Validation summary...

Training Set:
  Shape: (938088, 77)
  Mean: 0.000000
  Std: 1.000000
  Min: -0.122544
  Max: 51.289403

Test Set:
  Shape: (66221, 77)
  Mean: -0.002147
  Std: 1.043287
  Min: -0.122544
  Max: 4.914385

======================================================================
✅ DATA PREPROCESSING COMPLETE
======================================================================
```

**File 02 Deliverables:**
models/
├── label_encoder.joblib          (12 class labels)
└── feature_scaler.joblib          (StandardScaler parameters)
data/
├── X_train_scaled.npy             (938,088 × 77)
├── X_test_scaled.npy              (66,221 × 77)
├── y_train.npy                    (938,088 labels)
├── y_test.npy                     (66,221 labels)
└── feature_metadata.json          (feature names + scaler info)
**Key Takeaways from File 02:**

✅ **Feature Engineering**: 23 bidirectional features created  
✅ **Class Balance**: SMOTE oversampling achieved 1:1 ratio  
✅ **Scaling**: StandardScaler normalized to mean=0, std=1  
✅ **Stratified Split**: 80/20 train/test preserving class ratios  
✅ **Artifacts Saved**: All models and data ready for training  
✅ **Quality Validated**: No inf/NaN, consistent scaling  

---

# **END OF PART 1**

