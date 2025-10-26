# Zeek Network Monitoring Integration

[![Zeek](https://img.shields.io/badge/Zeek-6.0.3-00549E.svg)](https://zeek.org/)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04_LTS-E95420.svg)](https://ubuntu.com/)
[![Cluster](https://img.shields.io/badge/Architecture-Cluster_Mode-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Production-success.svg)]()

> Enterprise-grade network behavior analytics using Zeek cluster architecture for comprehensive protocol analysis and threat detection in virtualized environments.

## 🎯 Project Overview

This project documents the deployment of Zeek (formerly Bro) network analysis framework in cluster mode, providing real-time behavioral analytics and protocol-level traffic inspection across virtualized network segments. The implementation demonstrates professional problem-solving by honestly documenting monitoring scope and identifying appropriate tools for different network zones.

### Key Innovation

**Challenge**: Physical attack box (Kali) to virtual target monitoring requires different tooling approach than VM-to-VM monitoring.

**Solution**: Deploy Zeek for behavioral analytics on directly accessible network segments, document DMZ monitoring gap for signature-based IDS/IPS (Snort - Project 10).

### Key Features

- **Cluster Architecture**: Manager, proxy, and dual worker processes
- **Multi-Interface Monitoring**: Two network segments simultaneously
- **Protocol Analysis**: Deep inspection of HTTP, DNS, SSL, SSH, SMTP
- **Behavioral Analytics**: Connection metadata and anomaly detection
- **SIEM-Ready Logs**: Structured logs for security platform integration
- **67% Coverage**: Internal Network + User-LAN (honest scope documentation)

## 📊 Architecture & Coverage

### Zeek Cluster Architecture

```
┌─────────────────────────────────────────────────┐
│         Zeek VM (192.168.30.80)                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  Manager:    Centralized coordination           │
│  Proxy-1:    Load balancing and filtering       │
│  Worker-1:   Monitors enp0s3 (30.x Internal)    │
│  Worker-2:   Monitors enp0s8 (40.x User-LAN)    │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Network Coverage

| Network Segment | Coverage | Method | Status |
|-----------------|----------|--------|--------|
| **Internal Network** (192.168.30.0/24) | 100% | Worker-1 (enp0s3) | ✅ Operational |
| **User-LAN** (192.168.40.0/24) | 100% | Worker-2 (enp0s8) | ✅ Operational |
| **Management Network** (192.168.10.0/24) | 0% | Not Applicable | ⚠️ Documented Limitation |

**Monitored VMs**: Wazuh, ELK Stack, MISP, OpenCTI, Shuffle, Windows Server, DVWA, Metasploitable

**Excluded**: Physical Kali attack box (192.168.10.101) - requires different monitoring approach (Snort IDS/IPS)

### Network Topology

```
Internet → Router (192.168.2.0/24)
    ↓
MAC-MINI Physical Host
├── pfSense A VM (192.168.10.1)
│   └── Management Network (192.168.10.0/24)
│       └── Physical Kali (192.168.10.101) ⚠️ Not Monitored
└── Pi-hole DNS (192.168.10.106)
    ↓
Desktop Host (Windows) - VirtualBox Hypervisor
├── pfSense B VM (192.168.30.1 / 192.168.40.1)
├── Zeek VM (192.168.30.80) ⭐ Monitoring Host
│   ├── enp0s3: Internal Network (promiscuous)
│   └── enp0s8: User-LAN (promiscuous)
│
├── Internal Network (192.168.30.0/24) ✅ Monitored
│   ├── Wazuh SIEM (192.168.30.10)
│   ├── ELK Stack (192.168.30.20)
│   ├── Windows Server (192.168.30.40)
│   ├── MISP (192.168.30.50)
│   ├── OpenCTI (192.168.30.60)
│   ├── Shuffle SOAR (192.168.30.70)
│   └── DVWA (192.168.30.90)
│
└── User-LAN (192.168.40.0/24) ✅ Monitored
    └── Metasploitable (192.168.40.10)
```

## 🚀 Quick Start

### Prerequisites

**VM Specifications:**
- OS: Ubuntu 22.04 LTS
- RAM: 4 GB minimum
- CPU: 2 cores minimum
- Disk: 50 GB (for logs with rotation)
- Network: 2 interfaces on VirtualBox internal networks

**Required Software:**
- cmake (3.28.3+)
- gcc/g++ compiler toolchain
- libpcap-dev, libssl-dev
- python3-dev, swig, zlib1g-dev

### Installation Steps

#### 1. Create Zeek VM in VirtualBox

```powershell
# On Windows Host - Create VM
VBoxManage createvm --name "Zeek" --ostype Ubuntu_64 --register
VBoxManage modifyvm "Zeek" --memory 4096 --cpus 2
VBoxManage createhd --filename "Zeek.vdi" --size 51200

# Configure first network adapter (Internal Network)
VBoxManage modifyvm "Zeek" --nic1 intnet
VBoxManage modifyvm "Zeek" --intnet1 "Internal Network"
VBoxManage modifyvm "Zeek" --nicpromisc1 allow-all

# Install Ubuntu 22.04 LTS
```

#### 2. Install Build Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y cmake make gcc g++ flex bison libpcap-dev \
  libssl-dev python3 python3-dev swig zlib1g-dev \
  libmaxminddb-dev git curl

# Verify installations
cmake --version    # Should show 3.28.3+
gcc --version      # Should show 13.3.0+
python3 --version  # Should show 3.12.3+
```

#### 3. Compile Zeek from Source

```bash
# Download Zeek source
cd ~
wget https://download.zeek.org/zeek-6.0.3.tar.gz
tar -xzf zeek-6.0.3.tar.gz
cd zeek-6.0.3

# Configure (install to /opt/zeek)
./configure --prefix=/opt/zeek

# Compile (takes 15-30 minutes)
make -j2

# Install
sudo make install
```

#### 4. Configure Environment

```bash
# Add Zeek to PATH
echo 'export PATH=/opt/zeek/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
zeek --version
# Output: zeek version 6.0.3
```

#### 5. Add Second Network Interface

```powershell
# On Windows Host - Shutdown VM and add adapter
VBoxManage controlvm "Zeek" poweroff

# Add UserLAN network adapter
VBoxManage modifyvm "Zeek" --nic2 intnet
VBoxManage modifyvm "Zeek" --intnet2 "UserLAN Network"
VBoxManage modifyvm "Zeek" --nicpromisc2 allow-all

# Start VM
VBoxManage startvm "Zeek"
```

#### 6. Configure Second Interface

```bash
# On Zeek VM - Bring up second interface
sudo ip link set enp0s8 up

# Make permanent via netplan
sudo nano /etc/netplan/00-installer-config.yaml
```

Add:
```yaml
network:
  version: 2
  ethernets:
    enp0s3:
      dhcp4: true
    enp0s8:
      dhcp4: false  # No IP needed for monitoring-only interface
```

Apply:
```bash
sudo netplan apply

# Verify both interfaces UP
ip addr show | grep "state UP"
```

#### 7. Define Network Topology

```bash
# Edit networks configuration
sudo nano /opt/zeek/etc/networks.cfg
```

Add:
```ini
# Management Network (NOT monitored by this Zeek instance)
192.168.10.0/24    Management

# Internal Network (monitored via enp0s3)
192.168.30.0/24    Internal

# User-LAN (monitored via enp0s8)
192.168.40.0/24    User-LAN
```

#### 8. Configure Cluster Mode

```bash
# Edit node configuration
sudo nano /opt/zeek/etc/node.cfg
```

Replace with:
```ini
[manager]
type=manager
host=localhost

[proxy-1]
type=proxy
host=localhost

[worker-1]
type=worker
host=localhost
interface=enp0s3

[worker-2]
type=worker
host=localhost
interface=enp0s8
```

#### 9. Deploy Zeek Cluster

```bash
# Install configuration
sudo /opt/zeek/bin/zeekctl install

# Deploy Zeek
sudo /opt/zeek/bin/zeekctl deploy

# Check status
sudo /opt/zeek/bin/zeekctl status
```

Expected output:
```
Name       Type      Host       Status    Pid    Started
manager    manager   localhost  running   1938   30 Sep 06:24:30
proxy-1    proxy     localhost  running   1988   30 Sep 06:24:31
worker-1   worker    localhost  running   2055   30 Sep 06:24:33
worker-2   worker    localhost  running   2056   30 Sep 06:24:33
```

#### 10. Verify Log Generation

```bash
# Check logs directory
ls -lh /opt/zeek/logs/current/

# View connection log
sudo /opt/zeek/bin/zeek-cut id.orig_h id.resp_h id.resp_p < /opt/zeek/logs/current/conn.log | tail -10

# View DNS queries
sudo /opt/zeek/bin/zeek-cut query answers < /opt/zeek/logs/current/dns.log | tail -10
```

📖 For detailed implementation, troubleshooting, and architectural analysis, refer to the [Complete Implementation Guide](IMPLEMENTATION.md).

## 🔧 Usage & Management

### Zeek Service Management

```bash
# Check cluster status
sudo /opt/zeek/bin/zeekctl status

# Restart all components
sudo /opt/zeek/bin/zeekctl restart

# Stop Zeek
sudo /opt/zeek/bin/zeekctl stop

# Start Zeek
sudo /opt/zeek/bin/zeekctl start

# View diagnostics (when crashed)
sudo /opt/zeek/bin/zeekctl diag
```

### Log Analysis Commands

**Connection Analysis:**
```bash
# View recent connections
sudo /opt/zeek/bin/zeek-cut id.orig_h id.resp_h id.resp_p service < /opt/zeek/logs/current/conn.log | tail -20

# Find connections to specific host
sudo /opt/zeek/bin/zeek-cut id.orig_h id.resp_h id.resp_p < /opt/zeek/logs/current/conn.log | grep "192.168.40.10"
```

**DNS Analysis:**
```bash
# View DNS queries
sudo /opt/zeek/bin/zeek-cut query qtype_name answers < /opt/zeek/logs/current/dns.log | tail -20

# Find queries for specific domain
sudo /opt/zeek/bin/zeek-cut query answers < /opt/zeek/logs/current/dns.log | grep "google.com"
```

**Protocol Anomalies:**
```bash
# View protocol violations
sudo cat /opt/zeek/logs/current/weird.log

# View security notices
sudo cat /opt/zeek/logs/current/notice.log
```

**Service Discovery:**
```bash
# View discovered services
sudo cat /opt/zeek/logs/current/known_services.log
```

## 📊 Testing & Validation

### Internal Network Traffic Test

```bash
# From Zeek VM
ping -c 3 192.168.30.10  # Ping Wazuh
nslookup google.com      # DNS query

# Wait for processing
sleep 30

# View captured traffic
sudo /opt/zeek/bin/zeek-cut id.orig_h id.resp_h id.resp_p < /opt/zeek/logs/current/conn.log | tail -5
sudo /opt/zeek/bin/zeek-cut query answers < /opt/zeek/logs/current/dns.log | tail -5
```

### Cross-Network Attack Simulation

```bash
# From any VM on Internal Network (e.g., OpenCTI at 192.168.30.60)
nmap -p 22,80,443,3306 192.168.40.10  # Scan Metasploitable

# On Zeek VM - View captured scan
sudo /opt/zeek/bin/zeek-cut id.orig_h id.resp_h id.resp_p service < /opt/zeek/logs/current/conn.log | grep "192.168.40.10"
```

Expected output:
```
192.168.30.60  192.168.40.10  22    ssh
192.168.30.60  192.168.40.10  80    http
192.168.30.60  192.168.40.10  3306  mysql
192.168.30.60  192.168.40.10  445   smtp
```

### Generated Log Files

```bash
/opt/zeek/logs/current/
├── broker.log           # Inter-process communication
├── capture_loss.log     # Packet loss statistics
├── cluster.log          # Cluster management
├── conn.log             # Connection summaries ⭐
├── dns.log              # DNS queries/responses ⭐
├── http.log             # HTTP requests
├── known_services.log   # Discovered services
├── notice.log           # Security notices ⭐
├── ssl.log              # TLS/SSL connections
├── stats.log            # Resource usage
└── weird.log            # Protocol anomalies ⭐
```

## 🛠️ Troubleshooting

### Issue 1: Zeek Components Crashed

**Symptoms:**
```bash
sudo /opt/zeek/bin/zeekctl status
# Output: worker-1  worker  localhost  crashed  [fatal_error]
```

**Solution:**
```bash
# View diagnostic information
sudo /opt/zeek/bin/zeekctl diag

# Check error logs
sudo cat /opt/zeek/logs/current/stderr.log

# Restart services
sudo /opt/zeek/bin/zeekctl restart

# Verify all running
sudo /opt/zeek/bin/zeekctl status
```

### Issue 2: Interface Not Capturing Traffic

**Symptoms:**
- Empty conn.log or dns.log
- No traffic being captured

**Solution:**
```bash
# Verify interface is UP
ip addr show enp0s3
ip addr show enp0s8

# Check promiscuous mode enabled
ip link show enp0s3 | grep PROMISC
ip link show enp0s8 | grep PROMISC

# Verify VirtualBox promiscuous mode
# On Windows Host:
VBoxManage showvminfo "Zeek" | findstr promisc
# Should show: allow-all

# Generate test traffic to verify capture
ping 192.168.30.10
```

### Issue 3: Configuration Syntax Errors

**Symptoms:**
```
fatal error: problem with interface enp0s3,enp0s8
```

**Solution:**
```bash
# Verify node.cfg syntax
sudo cat /opt/zeek/etc/node.cfg

# Each worker must have ONE interface
# ❌ Wrong: interface=enp0s3,enp0s8
# ✅ Correct: Separate workers per interface

[worker-1]
type=worker
host=localhost
interface=enp0s3

[worker-2]
type=worker
host=localhost
interface=enp0s8
```

### Issue 4: Compilation Failures

**Symptoms:**
- Source compilation errors
- Missing dependencies

**Solution:**
```bash
# Install all build dependencies
sudo apt install -y cmake make gcc g++ flex bison \
  libpcap-dev libssl-dev python3 python3-dev swig \
  zlib1g-dev libmaxminddb-dev

# Clean and retry
cd ~/zeek-6.0.3
make clean
./configure --prefix=/opt/zeek
make -j2
sudo make install
```

### Diagnostic Commands

```bash
# Check system resources
free -h
df -h
top

# Verify network interfaces
ip addr show
ip link show

# Test packet capture manually
sudo tcpdump -i enp0s3 -c 10
sudo tcpdump -i enp0s8 -c 10

# View Zeek process status
ps aux | grep zeek

# Check log file permissions
ls -lh /opt/zeek/logs/current/
```

## 💡 Architectural Insights

### Why Cluster Mode?

**Standalone Mode Limitation:**
```bash
# ❌ This doesn't work:
zeek -i enp0s3,enp0s8  # Interprets as single interface name
```

**Cluster Mode Solution:**
```
✅ Manager:  Centralized coordination
✅ Proxy:    Load balancing and filtering
✅ Worker-1: Dedicated to enp0s3 (Internal)
✅ Worker-2: Dedicated to enp0s8 (User-LAN)
```

### Physical/Virtual Monitoring Challenge

**Attempted & Failed Approaches:**

1. **Bridged Adapter** - Different physical hosts, can't see traffic
2. **pfSense Promiscuous Mode** - Layer 3 routing bypasses Layer 2 capture
3. **Static Routes** - Routing works, but Zeek can't capture routed traffic
4. **Move DVWA** - Connectivity established, but no packet visibility

**Root Cause:**
```
Physical Kali (192.168.10.101)
    ↓ Routes through MAC-MINI pfSense
    ↓ Then through Desktop pfSense B
    ↓ To DVWA (192.168.30.90)

Zeek Position: Same network as DVWA (192.168.30.80)
Issue: Routed traffic (Layer 3) doesn't generate
       broadcast frames for promiscuous capture (Layer 2)
```

**Architectural Realization:**
- **Zeek**: Behavioral analytics on directly accessible networks ✅
- **Snort IDS/IPS**: Signature-based inline at routing chokepoints ✅
- **Solution**: Use right tool for each zone (multi-tool strategy)

### Coverage Decision

| Network | Tool | Reason |
|---------|------|--------|
| Internal (30.x) | Zeek | VM-to-VM traffic, behavioral analytics |
| User-LAN (40.x) | Zeek | VM-to-VM attacks, protocol analysis |
| DMZ (20.x) | Snort (Project 10) | Physical attacks, inline detection |

## 🎓 Skills Demonstrated

### Technical Competencies

**Network Monitoring**
- Zeek cluster deployment and configuration
- Multi-interface packet capture strategies
- VirtualBox promiscuous mode networking
- Protocol analysis and service identification

**System Administration**
- Linux source compilation (cmake, make)
- Dependency management and troubleshooting
- Service crash recovery and diagnostics
- Performance monitoring and optimization

**Security Analysis**
- Behavioral analytics and anomaly detection
- Protocol analysis (DNS, HTTP, SSH, SMTP)
- Attack traffic correlation
- Network baseline establishment

**Problem-Solving**
- Architecture constraint identification
- Multi-tool strategy development
- Honest scope documentation
- Systematic troubleshooting methodology


## 📈 Performance Metrics

### Implementation Results

| Metric | Value |
|--------|-------|
| **Configuration Time** | 6 hours (compilation + config) |
| **Troubleshooting Time** | 2 hours (interfaces + crashes) |
| **Total Project Time** | 10 hours |
| **Cluster Deployment Success** | 100% (4/4 components running) |
| **Network Coverage** | 67% (2/3 segments) |
| **Monitored VMs** | 8 systems |
| **Protocol Detection** | Active (DNS, HTTP, SSH, MySQL, SMTP) |

### Resource Usage

```
Per Worker Process:
- CPU: 10-15%
- RAM: ~500 MB
- Disk I/O: Moderate (continuous logging)

Total Zeek VM:
- RAM: 4 GB allocated
- CPU: 2 cores
- Disk: 50 GB (with log rotation)
```

## 🔄 Next Steps

### Immediate Enhancements (0-2 weeks)
- [ ] Develop custom Zeek scripts for specific attacks
- [ ] Configure log rotation policies
- [ ] Create baseline traffic profiles
- [ ] Document common analysis queries

### Intermediate Expansion (2-3 months)
- [ ] Integrate Zeek logs with ELK Stack (Week 3)
- [ ] Automated alerting for suspicious connections
- [ ] Detection rules for MITRE ATT&CK techniques
- [ ] File extraction for malware analysis

### Long-term Integration (3-6 months)
- [ ] Full SIEM correlation (Wazuh + ELK + Zeek)
- [ ] Automated threat hunting workflows
- [ ] MISP integration for IOC enrichment
- [ ] ML-based anomaly detection

### Project 10: DMZ Monitoring (Deferred)
- [ ] Deploy Snort IDS/IPS on MAC-MINI pfSense
- [ ] Add DMZ interface (192.168.20.0/24)
- [ ] Monitor physical Kali → DMZ attacks
- [ ] Complement Zeek with signature-based detection

## 📚 Key Learnings

### Tool-Appropriate Deployment

```yaml
Success Factors:
  ✅ Zeek excels at behavioral analytics
  ✅ VM-to-VM traffic fully visible
  ✅ Protocol analysis comprehensive
  ✅ SIEM integration ready

Honest Limitations:
  ⚠️ Physical/virtual integration requires different tools
  ⚠️ Layer 3 routing bypasses Layer 2 capture
  ⚠️ DMZ monitoring better suited for Snort
  ⚠️ Multi-tool strategy for enterprise coverage
```

### Professional Documentation

**Key Principle**: Document what you *didn't* accomplish and why, not just successes.

```
❌ Bad: "Achieved 100% coverage" (false)
✅ Good: "67% coverage (Internal + User-LAN), DMZ monitoring 
         deferred to Project 10 (Snort) due to Layer 3 routing
         constraints preventing promiscuous capture"
```

This demonstrates:
- Honest assessment
- Understanding of limitations
- Proper tool selection
- Professional maturity

## 📖 References

### Official Documentation
- [Zeek Documentation](https://docs.zeek.org/)
- [Zeek Cluster Configuration](https://docs.zeek.org/en/master/cluster/)
- [Zeek Log Analysis](https://docs.zeek.org/en/master/logs/)
- [VirtualBox Networking](https://www.virtualbox.org/manual/ch06.html)

### Technical Resources
- [Zeek GitHub Repository](https://github.com/zeek/zeek)
- [Zeek Package Manager](https://packages.zeek.org/)
- [Zeek Community](https://community.zeek.org/)
- [Zeek Training](https://zeek.org/training/)

### Standards & Frameworks
- [MITRE ATT&CK](https://attack.mitre.org/) - Network-based detection
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [PCAP File Format](https://www.tcpdump.org/pcap.html)

## 👤 Author

**Prageeth Panicker**

- GitHub: [@pragepani](https://github.com/pragepani)
- LinkedIn: [LinkedIn](https://www.linkedin.com/in/prageeth-panicker)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Zeek team for excellent network analysis framework
- VirtualBox community for virtualization platform
- Network security community for best practices
- Cybersecurity home lab community

---

**Project Status**: ✅ Implemented with Documented Limitations  
**Coverage**: 67% (2/3 Network Segments)  
**Last Updated**: October 24, 2025  
**Deployment Date**: September 30, 2025

---

*Part of a comprehensive cybersecurity home lab project series. This project demonstrates professional problem-solving through honest scope documentation and appropriate tool selection for different monitoring scenarios.*
