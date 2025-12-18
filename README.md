# 🔐 Cybersecurity Portfolio 

Welcome!  
This repository hosts my **professional portfolio website** documenting real-world, production-grade **Cybersecurity projects** in home lab setup.

🌐 **Live site:** (https://github.com/pragepani/Cybersecurity)


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
| 6 | DNS Abuse & Exfiltration Simulation | ✅ Completed | Detection of DNS tunneling & DNS security limitations |
| 7 | Wazuh Agent Fleet Deployment | ✅ Completed | EDR-style visibility across lab systems |
| 8 | ELK Stack SIEM Implementation | ✅ Completed | Centralized log analytics (Wazuh, Zeek, pfSense) |
| 9 | SIEM Data Quality Sprint | ✅ Completed | Parsing, normalization & 99.5% noise reduction |
| 10 | Snort IDS/IPS Integration | ✅ Completed | Perimeter IDS/IPS with WAN blocking & LAN monitoring |

**Foundation Result:** Production-grade, multi-layer monitoring stack (pfSense + Wazuh + Zeek + ELK + Snort + Pi-hole) operational across LAN/DMZ-style segments.

---

### 🧠 Completed AI / Detection Systems

| # | AI / Lab Project | Status | Goal / Focus |
|:-:|-------------------|:-------:|--------------|
| AI-1 | **NIDS-ML Development** (Part 1 & 2) | ✅ Completed | Network Intrusion Detection System with LightGBM + Autoencoder + GenAI explainer |

---

## 🧩 Technical Highlights

- **Hybrid NIDS** → LightGBM + Autoencoder ensemble validated on CIC-IDS2017  
- **Explainability Tiering** → Cache → Rules → LLM fallback for zero false negatives  
- **DNS Security Work** → Demonstrated DNS tunneling bypass plus documented architectural limits in pfSense DNS enforcement  
- **Malware / Threat Visibility** → Zeek protocol analytics + Wazuh host telemetry + Snort signature detection  
- **Threat Intel & SIEM** → Wazuh + Zeek + pfSense logs normalized into ELK with ILM + dashboards  
- **Noise Reduction** → 99.5% reduction of pfSense noise (DNS/NTP) while preserving 100% security-relevant events  

---

## 🧠 Technologies & Tools

**Security Stack:** pfSense · Snort · Zeek · ELK · Wazuh · MISP · OpenCTI · Shuffle · Pi-hole  
**Infra & Dev:** Linux  
**Visualization:** Plotly · Seaborn  
**Frameworks:** MITRE ATT&CK · MISP API · SOAR Playbooks  
**AI/ML:** Python · scikit-learn · LightGBM · LLM · Regression Models · Classification Models · NLP  

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

> “Quality over speed — every project is a portfolio piece.”  

---

### 🪪 License
Content © Prageeth Panicker · 2025  
Source code under MIT License unless otherwise specified.
