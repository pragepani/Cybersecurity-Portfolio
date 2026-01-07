# Enterprise Network Security Architecture (Cybersecurity Capstone)

**Author:** Prageeth Panicker  
**Timeframe:** Nov–Dec 2025  
**Project type:** End-to-end design, build, security implementation, and validation of a segmented enterprise network for a mid-sized professional services firm (100–175 users).

This repository captures a complete, lab-based **enterprise network + security monitoring build** aligned to **NIST CSF 2.0** and **Zero Trust** principles, including **high availability**, **VLAN segmentation**, **Active Directory**, **SIEM/EDR**, **vulnerability management**, and **SOC workflow execution**.

---

## What you’ll find here

- **Stage 1 — Network Plan**: requirements, design principles, topology goals, inventory/capacity planning, compliance mapping
- **Stage 2 — Build & Configuration**: VLAN/IP plan, switching, pfSense HA (CARP/pfsync), DHCP relay, server IP allocations, known issues
- **Stage 3 — Security Planning**: threat model + risk analysis, firewall philosophy, hardening checklist, monitoring & SOC processes, implementation roadmap
- **Stage 4 — Security Implementation & Validation**: Wazuh + Splunk SIEM, alerting, dashboards, SOC incident reports, Nessus scanning, remediation & re-validation

> Recommended: read the PDFs in `docs/` in order: Segment1 → Segment2 → Segment3 → Segment4.

---

## Architecture overview

### Platform & environment
- **Lab platform:** EVE-NG on **VMware Workstation Pro**
- **Topology:** **Dual pfSense firewalls (CARP HA)** + **4-switch ring** with **HSRP** on management VLAN
- **VLAN design:** **16 VLANs designed** (15 used in implementation; one reserved for future growth)
- **Internet links (design):** ISP Fiber + 5G failover (dual ISP concept)

### Security tooling (implementation)
- **Wazuh** (EDR / log collection / FIM / alerting)
- **Splunk** (SIEM searches + dashboards)
- **Nessus** (vulnerability scanning + remediation validation)

---

## Network segmentation (VLANs)

|   VLAN | Name             | Purpose                                    | Subnet       | Gateway/VIP           |
|-------:|:-----------------|:-------------------------------------------|:-------------|:----------------------|
|     10 | Management       | Network device management                  | 10.0.0.0/24  | 10.0.0.119 (HSRP VIP) |
|     20 | Corporate        | Reserved for future use                    | 10.20.0.0/25 | 10.20.0.126           |
|     21 | Project Delivery | Marketing/Project team                     | 10.21.0.0/25 | 10.21.0.126           |
|     22 | Sales            | Sales department                           | 10.22.0.0/25 | 10.22.0.126           |
|     23 | Engineering      | Engineering team                           | 10.23.0.0/25 | 10.23.0.126           |
|     24 | Finance          | Finance department                         | 10.24.0.0/24 | 10.24.0.254           |
|     25 | HR               | Human Resources                            | 10.25.0.0/25 | 10.25.0.126           |
|     26 | IT/Security      | IT staff & security team                   | 10.26.0.0/25 | 10.26.0.126           |
|     27 | Executive        | Executive leadership                       | 10.27.0.0/26 | 10.27.0.62            |
|     30 | Servers          | Internal server infrastructure             | 10.30.0.0/23 | 10.30.0.254           |
|     40 | Printers         | Network printing services                  | 10.40.0.0/24 | 10.40.0.254           |
|     50 | Guest WiFi       | Guest wireless access                      | 10.50.0.0/23 | 10.50.0.254           |
|     60 | IoT Devices      | Internet of Things devices                 | 10.60.0.0/24 | 10.60.0.254           |
|     70 | VoIP             | Voice over IP services                     | 10.70.0.0/24 | 10.70.0.254           |
|     80 | Security         | Security systems (cameras, access control) | 10.80.0.0/24 | 10.80.0.254           |
|     90 | Backup/Storage   | Backup and storage network                 | 10.90.0.0/24 | 10.90.0.254           |

> **Naming note:** VLAN30 is labeled as `"Guest"` in some switch VLAN databases (legacy naming artifact). Functionally VLAN30 is the **Server VLAN** hosting DC01/FILE-SRV01/APP-SRV01.

---

## IP allocations

### Server IP allocations
| Device      | IP            |   VLAN | Function                                  |
|:------------|:--------------|-------:|:------------------------------------------|
| DC01        | 10.30.0.10/23 |     30 | Domain Controller, DNS, DHCP              |
| FILE-SRV01  | 10.30.0.20/23 |     30 | File Server (SMB shares)                  |
| APP-SRV01   | 10.30.0.30/23 |     30 | Application Server (IIS, Employee Portal) |
| Jump-Server | 10.26.0.9/25  |     26 | Bastion host for secure management access |

### Core / management IP allocations
| Device           | Interface   | IP            | Purpose                             |
|:-----------------|:------------|:--------------|:------------------------------------|
| Core-SW1         | VLAN10      | 10.0.0.120/24 | Management (HSRP Priority 150)      |
| Core-SW2         | VLAN10      | 10.0.0.121/24 | Management (HSRP Priority 140)      |
| HSRP VIP         | VLAN10      | 10.0.0.119/24 | Virtual gateway for management VLAN |
| pfSense-FW       | VLAN10      | 10.0.0.1/24   | Primary firewall management         |
| pfSense-Failover | VLAN10      | 10.0.0.2/24   | Secondary firewall management       |

### pfSense interfaces (Primary/Secondary + VIPs)
| Interface   | VLAN   | Primary IP        | Secondary IP      | VIP               |
|:------------|:-------|:------------------|:------------------|:------------------|
| WAN         | -      | 192.168.60.130/24 | 192.168.60.131/24 | 192.168.60.129/24 |
| SYNC        | -      | 192.168.250.1/24  | 192.168.250.2/24  | N/A (dedicated)   |
| VLAN10      | 10     | 10.0.0.1/24       | 10.0.0.2/24       | 10.0.0.254/24     |
| VLAN20      | 20     | 10.20.0.1/25      | 10.20.0.2/25      | 10.20.0.126/25    |
| VLAN21      | 21     | 10.21.0.1/25      | 10.21.0.2/25      | 10.21.0.126/25    |
| VLAN22      | 22     | 10.22.0.1/25      | 10.22.0.2/25      | 10.22.0.126/25    |
| VLAN23      | 23     | 10.23.0.1/25      | 10.23.0.2/25      | 10.23.0.126/25    |
| VLAN24      | 24     | 10.24.0.1/24      | 10.24.0.2/24      | 10.24.0.254/24    |
| VLAN25      | 25     | 10.25.0.1/25      | 10.25.0.2/25      | 10.25.0.126/25    |
| VLAN26      | 26     | 10.26.0.1/25      | 10.26.0.2/25      | 10.26.0.126/25    |
| VLAN27      | 27     | 10.27.0.1/26      | 10.27.0.2/26      | 10.27.0.62/26     |
| VLAN30      | 30     | 10.30.0.1/23      | 10.30.0.2/23      | 10.30.0.254/23    |

### CARP VIP plan (default gateways per VLAN)
|   VLAN | CARP VIP    |   VHID | Primary IP   | Secondary IP   | Purpose              |
|-------:|:------------|-------:|:-------------|:---------------|:---------------------|
|     10 | 10.0.0.254  |      1 | 10.0.0.1     | 10.0.0.2       | Management           |
|     20 | 10.20.0.126 |      2 | 10.20.0.1    | 10.20.0.2      | Corporate (Reserved) |
|     21 | 10.21.0.126 |     21 | 10.21.0.1    | 10.21.0.2      | Project Delivery     |
|     22 | 10.22.0.126 |     22 | 10.22.0.1    | 10.22.0.2      | Sales                |
|     23 | 10.23.0.126 |     23 | 10.23.0.1    | 10.23.0.2      | Engineering          |
|     24 | 10.24.0.254 |     24 | 10.24.0.1    | 10.24.0.2      | Finance              |
|     25 | 10.25.0.126 |     25 | 10.25.0.1    | 10.25.0.2      | HR                   |
|     26 | 10.26.0.126 |     26 | 10.26.0.1    | 10.26.0.2      | IT/Security          |
|     27 | 10.27.0.62  |     27 | 10.27.0.1    | 10.27.0.2      | Executive            |
|     30 | 10.30.0.254 |      3 | 10.30.0.1    | 10.30.0.2      | Servers              |
|     40 | 10.40.0.254 |      4 | 10.40.0.1    | 10.40.0.2      | Printers             |
|     50 | 10.50.0.254 |      5 | 10.50.0.1    | 10.50.0.2      | Guest WiFi           |
|     60 | 10.60.0.254 |      6 | 10.60.0.1    | 10.60.0.2      | IoT Devices          |
|     70 | 10.70.0.254 |      7 | 10.70.0.1    | 10.70.0.2      | VoIP                 |
|     80 | 10.80.0.254 |      8 | 10.80.0.1    | 10.80.0.2      | Security Systems     |
|     90 | 10.90.0.254 |      9 | 10.90.0.1    | 10.90.0.2      | Backup/Storage       |

---

## Key components

### Switching layer (Core + Access)
- **Core switching design:** Layer-2 only (routing handled by pfSense), PVST enabled
- **HSRP:** configured on **VLAN10** for management redundancy
- **Access switch port mapping (highlights):**
  - Access-SW-Servers ports mapped to DC01 / FILE-SRV01 / APP-SRV01 on VLAN30
  - Department access switch provides ports for IT and Sales clients + trunks to core

> See `configs/switches/` for full configs and `docs/Segment2_*.pdf` for annotated build notes.

### Firewall layer (pfSense HA)
- **HA:** CARP VIPs for VLAN gateways + **pfsync** state synchronization over a **dedicated SYNC network (192.168.250.0/24)**
- **Design principle:** all inter-VLAN traffic is inspected (Zero Trust default deny)
- **DHCP relay:** handled on pfSense to forward client broadcasts to DC01 (DHCP server)

---

## Active Directory (AD) and core services

A Windows Server 2022 Domain Controller (**DC01**) provides centralized identity and core services:
- **Domain:** `company.local`
- **Roles/services:** **AD DS**, **DNS**, **DHCP**
- **DNS:** Forward zone `company.local`, reverse zone `30.10.in-addr.arpa`
- **DHCP design:** clients use DHCP `.100+`, servers are static `.10–.99`
  - Example scopes documented (Sales VLAN22, IT VLAN26) with exclusions and options
  - DHCP options: Router (003), DNS server (006 → DC01), DNS domain name (015 → `company.local`)

**AD OU structure (documented):**
- Sales, Engineering, Finance, HR, IT_Security, Executive, Servers, Workstations

---

## Security design principles

### Zero Trust, segmentation, and least privilege
- Default deny stance, explicit allow rules only
- Inter-VLAN access requires firewall inspection
- High-sensitivity zones include Finance, HR, Executive, Servers

### Hardening focus areas (documented + validated)
- Windows server posture hardening (e.g., SMB controls, audit policy)
- Linux hardening for admin access systems (SSH + host firewall)
- Governance baseline: CIS benchmark alignment (planned/implemented where applicable)

---

## Monitoring, alerting, and SOC workflows

### SIEM/EDR architecture
- Wazuh collects security telemetry (agents + manager)
- Logs and alerts searchable in Splunk (`index=wazuh ...`)
- Dashboards built for visibility (security overview + triage-focused panels)
- Alerts demonstrated for:
  - Failed authentication events
  - File Integrity Monitoring (FIM)
  - Service status changes

### SOC incident execution (Segment 4)
Complete incident reports are referenced as separate documents:
1. `SOC_Incident_Report_1_Failed_Logins.pdf`
2. `SOC_Incident_Report_2_File_Integrity_Monitoring.pdf`
3. `SOC_Incident_Report_3_Service_Status_Change.pdf`

Each report includes: executive summary, timeline, technical analysis, queries, root cause, MITRE mapping, controls validated, remediation, compliance impact, lessons learned.

---

## Vulnerability assessment & remediation

### Nessus scanning (Segment 4)
- Vulnerability assessment executed against core infrastructure and management systems
- Findings prioritized, remediated, then **post-remediation validated**

### Example remediation actions documented
- **SMB signing enforcement (Windows):** applied and verified on DC01 / FILE-SRV01 / APP-SRV01  
  - Server + client signing required; applied via **Group Policy** for persistence  
- **SSH hardening (Linux):** secure SSH settings applied (root login restrictions, key-based auth, disable password auth)
- **Host firewall (Linux):** UFW enabled with SSH allowed (default deny stance)

Accepted lab risk noted: self-signed certificates (lab environment).

---

## Validation and evidence

Segment 4 includes extensive screenshot-based evidence (SS-01 … SS-72) covering:
- HA and CARP status, VIP configuration, pfSync evidence
- Wazuh data verification in Splunk, agent health, index/retention
- Dashboard panels and alert views (failed auth, FIM, services)
- Nessus scan results + post-remediation proof (PowerShell/OS configs)


---

## How to reproduce (high-level)

1. **Build the lab platform**
   - Deploy EVE-NG under VMware Workstation Pro
   - Import the topology (pfSense HA + core/access switching + servers)

2. **Configure switching**
   - Create VLANs, trunks, PVST
   - Configure HSRP on VLAN10
   - Assign access ports per department/server roles

3. **Configure pfSense HA**
   - Create VLAN interfaces + IPs
   - Configure CARP VIPs, pfsync on SYNC network
   - NAT for outbound internet, baseline firewall policy
   - Enable DHCP relay to DC01 for client VLANs

4. **Deploy servers and AD**
   - Configure DC01 (AD DS/DNS/DHCP) and server roles (FILE-SRV01/APP-SRV01)
   - Join endpoints as needed

5. **Implement security monitoring**
   - Deploy Wazuh manager + agents
   - Configure Splunk inputs/indexing + dashboards
   - Run Nessus scans, remediate, and retest

6. **Run validation + incidents**
   - Execute validation checklist
   - Generate and document the 3 SOC incident scenarios

---

## Known issues / lessons learned (from build docs)

- VLAN30 naming inconsistency (“Guest” label on some switches) — cosmetic, planned cleanup
- A CARP inconsistency was observed for VLAN27 VIP state (“split-brain” symptom); root cause points to pfSense VIP configuration mismatch (password/interface/VHID alignment). VLAN27 had no active hosts during Segment 2, so impact was limited.

---

## Roadmap (next improvements)
- Fix VLAN naming consistency across switch VLAN databases
- Close remaining crypto gaps (e.g., fully disable TLS 1.0/1.1 across relevant systems where applicable)
- Add 802.1X for dynamic VLAN assignment (design-ready)
- Expand dashboards, false-positive tuning, and SOAR automation playbooks
- Extend monitoring to additional hosts and network telemetry sources

---

## Repository structure (recommended)

```
.
├── README.md
│   ├── Segment1_NetworkPlan_Prageeth_Panicker_v1.4.pdf
│   ├── Segment2_Build-Configuraion_Prageeth_Panicker_v6.pdf
│   ├── Segment3_Security_Planning_Document_v3.1.pdf
│   └── Segment4_Validation_Prageeth_Panicker_v1.2.pdf
│   ├── switches/
│   ├── pfsense/
│   └── servers/
    ├── SOC_Incident_Report_1_Failed_Logins.pdf
    ├── SOC_Incident_Report_2_File_Integrity_Monitoring.pdf
    └── SOC_Incident_Report_3_Service_Status_Change.pdf
```


