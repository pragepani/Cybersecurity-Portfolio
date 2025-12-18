Splunk Enterprise SIEM Deployment \& Network Security Monitoring



Enterprise-Grade Implementation Using pfSense HA, Cisco Switches \& Zero Trust Architecture

Author: Prageeth Panicker — December 2025



📄 Overview



This repository documents the full deployment of a production-style Splunk Enterprise SIEM integrated with:



A pfSense High-Availability (CARP) firewall cluster



Cisco Core \& Access Switches (SSH-hardened + ACL-protected)



A unified Zero Trust network architecture



Centralized logging across multiple VLANs and network layers



The environment is built on a multi-VLAN enterprise topology using EVE-NG virtual infrastructure, simulating real SOC and NOC operations.



This project demonstrates SIEM deployment, network security hardening, centralized log ingestion, event correlation, and hands-on operational security.



🧱 Architecture Summary

🔐 Network Devices



Splunk SIEM Server: 10.0.0.75



pfSense HA Cluster: 10.0.0.1 (Primary), 10.0.0.2 (Secondary)



Core-SW1: 10.0.0.120



Core-SW2: 10.0.0.121



Access-SW (Servers): 10.0.0.161



Access-SW (Department): 10.0.0.171



Jump Server: 10.0.0.50 (Bastion/Access Control)



🌐 VLAN Structure

VLAN	Name	Subnet	Purpose

10	Management	10.0.0.0/24	Security infra (SIEM, Jump Server, EDR, Nessus)

22	Sales	10.22.0.0/25	Sales workstations

23	Finance	10.23.0.0/25	Finance workstations

24	HR	10.24.0.0/25	HR workstations

26	IT	10.26.0.0/25	IT workstation/management

30	Servers	10.30.0.0/24	DC, File servers, Application servers

80	Storage	10.80.0.0/24	NAS/storage

📦 Project Deliverables

✔ Splunk Enterprise Deployment (Ubuntu 24.04 LTS)



Installed Splunk Enterprise 10.0.2+



Configured indexes: network, windows, linux, wazuh



Enabled UDP/514 (syslog) \& TCP/9997 (forwarders)



Configured static IP + boot services



Hardened Splunk access using pfSense rules



✔ pfSense Firewall Integration



Enabled and verified remote syslog forwarding



Forwarded system + firewall + DHCP logs to Splunk



Tested block/allow traffic with Splunk visibility



Configured logging for all VLAN interfaces



✔ Cisco Switch Integration



Implemented across 4 switches:



SSHv2 enforced



RSA 2048-bit key generation



Legacy SSH ciphers enabled for interoperability



ACL restricting SSH only from Jump Server



Syslog forwarding to Splunk (logging host 10.0.0.75)



Source-interface binding (logging source-interface Vlan10)



Security event logging tested



✔ Jump Server (Zero Trust Enforcement)



Single point of administrative access



SSH client configuration updated for Cisco algorithms



Event logging routed to Splunk



📊 Data Collected (First 24 Hours)



30,977+ pfSense firewall events



System + security logs from all Cisco switches



DHCP, SSH attempts, configuration changes



All logs indexed in near real-time



These numbers reflect the real operational behavior of an enterprise network.



🧪 Functional Validation

✅ pfSense Log Validation



Firewall block/pass events captured



DHCP and system events verified



Log parsing validated via Splunk searches



✅ Switch Logging Validation



Configuration changes detected



SSH logins from Jump Server recorded



Unauthorized SSH attempts logged + blocked (ACL)



✅ SIEM Health



Indexing rate monitored



Splunk health report verified (green state)



Search performance confirmed



🛡️ Security Best Practices Implemented

🔐 Zero Trust Access



All administrative actions via Jump Server



No direct device access from user VLANs



🔐 SSH Hardening



SSHv2 only



Limited retries / timeouts



ACL-based source restrictions



Local authentication w/ strong credentials



🔐 Network Security



VLAN segmentation



pfSense firewall enforcing strict inter-VLAN controls



Syslog centralization



90-day log retention



📘 Documentation



A full 45-page technical implementation guide is included in this repository, covering:



Full step-by-step deployment



Network topology \& diagrams



Testing methodology



Troubleshooting steps



Security hardening



Future roadmap



Source Reference: Splunk Deployment \& Configuration Guide (PDF) 



Splunk\_Deployment\_Configuration…



📈 Future Enhancements



Wazuh → Splunk pipeline (EDR analytics)



MITRE ATT\&CK detection rules



SSL/TLS for Splunk web



Syslog over TLS



Automated Splunk alerting for anomalies



Compliance dashboards (NIST, CIS, ISO 27001)



High availability: Splunk indexer/search head clustering



🧑‍💻 Tech Stack



Tools:

Splunk Enterprise 10.0.2, pfSense 2.7+, Cisco IOS 15.2, Ubuntu Server 24.04 LTS, EVE-NG

Frameworks Referenced:

NIST CSF 2.0, CIS Controls v8, MITRE ATT\&CK



⭐ About This Project



This project simulates a production-grade enterprise SOC environment and demonstrates:



SIEM engineering



Network security engineering



Firewall \& switch hardening



Zero Trust architecture application



Incident detection fundamentals



Documentation \& operational excellence



It serves as a robust portfolio asset for cybersecurity roles including:



SOC Analyst | Cybersecurity Engineer | SIEM Engineer | Network Security Analyst

