# 🔐 Cybersecurity Portfolio 

Welcome!  
This repository hosts my **professional portfolio website** documenting real-world, production-grade **Cybersecurity projects** in home lab setup.

🌐 **Live site:** (https://github.com/pragepani/)  
📄 **Reference:** “Cybersecurity Project Plan — v1.0 (Oct 2025)”

## 🖧 Network Topology 
![image_alt](https://github.com/pragepani/Cybersecurity/blob/f9bea1c8f67174014ab13d9c630fc9b9cf8512a0/Current_Network_Diagram_v6_16thOct.jpg)

---

## 🧠 Focus Areas (Full Project Roadmap)

### ✅ Completed Lab Foundation (1 – 10)

| # | Project | Status | Key Focus |
|:-:|----------|:-------:|-----------|
| 1 | pfSense Multi-VLAN Deployment | ✅ Completed | Secure network segmentation and routing |
| 2 | Baseline Firewall Policy & NAT | ✅ Completed | Rule-set hardening and NAT architecture |
| 3 | Pi-hole DNS Filtering | ✅ Completed | Ad / malware domain blocking |
| 4 | SPAN/TAP Configuration + TShark | ✅ Completed | Traffic capture and packet analysis |
| 5 | Zeek Network Monitoring Integration | ✅ Completed | Network telemetry and logging |
| 6 | DNS Abuse & Exfiltration Simulation | ✅ Completed | Detection of DNS tunneling |
| 7 | Wazuh Agent Fleet Deployment | ✅ Completed | EDR visibility across fleet |
| 8 | ELK Stack SIEM Implementation | ✅ Completed | Centralized log analytics |
| 9 | SIEM Data Quality Sprint | ✅ Completed | Field mapping & noise reduction |
| 10 | Snort / Suricata Integration | ✅ Completed | Inline IDS/IPS testing |

**Foundation Result:** Production-grade monitoring stack operational across LAN/DMZ.

---

### 🚧 Phase 1 – Detection Systems (Weeks 1–6)

| # | AI Project | Status | Goal / Focus |
|:-:|-------------|:-------:|--------------|
| AI-1 | **NIDS-ML Development** (Part 1 & 2) | 🚧 In Progress | Network Intrusion Detection System with LightGBM + Autoencoder + GenAI explainer |
| Lab 11–12 | **NIDS Validation + Red Team / WAF** | 🚧 In Progress | Real-world attack simulation and WAF rule tuning |
| AI-2 | **Phishing Classifier** (Part 1 & 2) | 🕓 Scheduled | NLP & LLM-based email threat detection |
| Lab 13–15 | **Threat Intelligence Stack (MISP + OpenCTI + SOAR)** | 🕓 Scheduled | Operational IOC enrichment and automation |

---

### 🧬 Phase 2 – Malware & Advanced Analytics (Weeks 7–12)

| # | AI Project | Status | Key Focus |
|:-:|-------------|:-------:|-----------|
| AI-3 | **Malware Detection & Explainability** | 🕓 Planned | Static PE analysis + SHAP visual explanations |
| Lab 16–17 | **AD Hardening & Patch Automation** | 🕓 Planned | Windows security baseline & automated updates |
| Lab 18–20 | **UEBA & Threat Hunting Playbooks** | 🕓 Planned | Behavioral analytics + lateral movement detection |
| AI-4 | **Threat Intel Analyzer** | 🕓 Planned | IOC clustering + MISP integration + LLM campaign summaries |

---

### 🧰 Phase 3 – Incident Response & SOC Dashboard (Weeks 13–18)

| # | Project | Status | Key Focus |
|:-:|----------|:-------:|-----------|
| Lab 21–23 | **Incident Response & Forensics** | 🕓 Planned | Memory forensics + IR playbooks |
| AI-5 | **SOC Dashboard & GenAI Copilot** | 🕓 Planned | Unified SOC visibility + LLM assistant |
| Lab 31–42 | **Advanced Labs (Cloud, Detection, Purple Team)** | 🕓 Future | Zero-Trust / Deception / Vulnerability Assessment |
| Week 18 | **Portfolio Documentation & Polish** | 🕓 Final Stage | Website + Diagrams + Blog + Presentation Deck |

---

## 🧩 Technical Highlights

- **Hybrid NIDS** → LightGBM + Autoencoder ensemble validated on CIC-IDS2017   
- **Explainability Tiering** → Cache → Rules → LLM fallback for zero false negatives   
- **Phishing Classifier** → TF-IDF n-grams + URL entropy + GenAI intent detection   
- **Malware Triager** → SHAP waterfall plots + LLM verdict summaries   
- **Threat Intel Analyzer** → HDBSCAN clustering + MISP campaign cards   
- **SOC Copilot Dashboard** → RAG queries + ATT&CK heatmaps + automated reports   

---

## 🧠 Technologies & Tools

**Security Stack:** pfSense · Snort · Zeek · ELK · Wazuh · MISP · OpenCTI · Shuffle · Pi-hole 
**Infra & Dev:** Linux 
**Visualization:** Plotly · Seaborn 
**Frameworks:** MITRE ATT&CK · MISP API · SOAR Playbooks 
**AI/ML:** Python · scikit-learn · LightGBM · LLM · Regression Models, Classification Model, NLP

---

## 📊 Key Metrics

| Category | Metric | Target | Status |
|-----------|---------|---------|---------|
| NIDS Model | Recall ≥ 95 % |  |  |
| Phishing Model | Precision ≥ 90 % |  |  |
| Malware Model | ROC-AUC ≥ 0.95 |  |  |
| SIEM Throughput | > 200 K events/day |  |  |
| LLM Copilot Latency | < 2 s |  |  |

---

## 🚀 Next Goals (Phase 4 — Polish & Showcase)

- Complete AI SOC Dashboard demo video  
- Integrate Threat Intel Analyzer into MISP workflow  
- Add Purple-Team continuous detection program  
- Publish 3–5 technical blog posts  
- Launch custom domain + PDF resume integration  

---

> “Quality over speed — every project is a portfolio piece.”  

---

### 🪪 License
Content © Prageeth Panicker · 2025  
Source code under MIT License unless otherwise specified.
