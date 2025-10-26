# pfSense Baseline Firewall Policy & NAT Configuration

[![pfSense Version](https://img.shields.io/badge/pfSense-2.8.0-blue.svg)](https://www.pfsense.org/)
[![NAT Type](https://img.shields.io/badge/NAT-Port%20Forwarding-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Production-success.svg)]()

> Implementation of Network Address Translation (NAT) and port forwarding for controlled external access to web application security testing environments.

## 🎯 Project Overview

This project documents the implementation of baseline NAT policies and controlled external access for the DVWA (Damn Vulnerable Web Application) server in a cybersecurity home lab. The configuration enables realistic external attack scenarios while maintaining proper network segmentation and security boundaries.

### Key Features

- **Port Forwarding**: WAN port 8080 → Internal DVWA service (192.168.10.20:80)
- **Controlled Access**: Router network access only (no Internet exposure)
- **Automatic NAT**: Outbound NAT with automatic rule generation
- **Associated Filter Rules**: Properly linked port forward and firewall rules
- **Security Testing**: External attack simulation capability for penetration testing

## 🏗️ Network Architecture

```
Router Network (192.168.2.0/24)
         │
         │ Testing Client
         │ (192.168.2.222)
         │
         └─── WAN: 192.168.2.229:8080
                   │
              pfSense (MAC-MINI)
                   │
              LAN: 192.168.10.1
                   │
         Management Network (192.168.10.0/24)
                   │
                   ├─── DVWA Server (192.168.10.20:80)
                   ├─── Kali Linux (192.168.10.20)
                   └─── Attack Systems
```

### Network Details

| Component | IP Address | Purpose | Port |
|-----------|------------|---------|------|
| pfSense WAN | 192.168.2.229 | External interface (DHCP) | 8080 (external) |
| pfSense LAN | 192.168.10.1 | Management network gateway | - |
| DVWA Server | 192.168.10.20 | Vulnerable web application | 80 (internal) |
| Testing Client | 192.168.2.222 | External test system | - |

## 🚀 Quick Start

### Prerequisites

- **pfSense**: Version 2.8.0 with NAT capabilities
- **DVWA Server**: Running on 192.168.10.20:80
- **Network Access**: Testing client on 192.168.2.0/24 network
- **Administrative Access**: pfSense web interface credentials

### Implementation Steps

1. **Verify Current NAT Configuration**
   ```bash
   # Access pfSense
   https://192.168.10.1
   
   # Navigate to: Firewall → NAT → Outbound
   # Verify: Automatic outbound NAT rule generation enabled
   ```

2. **Verify DVWA Service**
   ```bash
   # From internal network (192.168.10.x):
   curl http://192.168.10.20:80
   ```

3. **Configure Port Forward**
   - Navigate to: Firewall → NAT → Port Forward
   - Add rule with the following settings:
     - Interface: WAN
     - Protocol: TCP
     - Destination: WAN address
     - Destination Port: 8080
     - Redirect Target IP: 192.168.10.20
     - Redirect Target Port: 80
     - **Filter rule association: Pass** (Critical!)

4. **Test External Access**
   ```bash
   # From router network (192.168.2.x):
   curl http://192.168.2.229:8080
   
   # Or in browser:
   http://192.168.2.229:8080
   ```

📖 For detailed implementation steps and troubleshooting, refer to the [Complete Implementation Guide](IMPLEMENTATION.md).

## 🔧 Configuration Details

### NAT Configuration

#### Outbound NAT Rules
```
Mode: Automatic outbound NAT rule generation

Rule 1: ISAKMP - VPN traffic passthrough (UDP port 500)
Rule 2: General - All internal networks → WAN address translation
  Networks: 192.168.10.0/24, 192.168.30.0/24, 192.168.40.0/24
```

#### Port Forward Rule
```
Interface: WAN (em0)
Protocol: TCP
Source: Any (restricted to 192.168.2.0/24 in practice)
Destination: WAN address
Destination Port: 8080
Redirect Target: 192.168.10.20:80
Description: DVWA Web Application Access From WAN
Filter Association: Pass (automatic WAN rule creation)
```

### Associated Firewall Rules

| Interface | Action | Protocol | Source | Destination | Port | Description |
|-----------|--------|----------|--------|-------------|------|-------------|
| WAN | Pass | TCP | Any | WAN address | 8080 | Auto-generated port forward rule |

## 📊 Testing & Validation

### Test Results

✅ **Internal Access (Pre-existing)**
```bash
# From Management Network (192.168.10.x):
http://192.168.10.20:80 → ✓ Success
```

✅ **External Access (Post-implementation)**
```bash
# From Router Network (192.168.2.x):
http://192.168.2.229:8080 → ✓ Success
```

❌ **Internet Access (Verified Blocked)**
```bash
# From Internet:
http://[public-ip]:8080 → ✗ Blocked (as intended)
```

### Security Validation

**Access Control Matrix:**
```
Internet → DVWA:              ✗ Blocked (no configuration)
Router Network → DVWA:        ✓ Allowed (via port 8080)
Management Network → DVWA:    ✓ Direct access (port 80)
Other Networks → DVWA:        ✗ Blocked (default deny)
```

### Performance Metrics

- **Configuration Time**: 3 hours (including troubleshooting)
- **External Access Success Rate**: 100% post-implementation
- **Security Boundary Maintenance**: Verified (no unintended exposure)
- **Traffic Processing**: Active and logged

## 🛠️ Troubleshooting

### Common Issues & Solutions

#### Issue 1: Port Forward Not Working

**Symptoms:**
- External connection attempts time out
- No traffic logged on WAN interface

**Solution:**
```bash
# Check filter rule association
Firewall → NAT → Port Forward → Edit rule
Verify: "Filter rule association" = "Pass" (not "None")

# pfSense must auto-create the associated WAN firewall rule
```

#### Issue 2: Destination Address Misconfiguration

**Symptoms:**
- Port forward configured but connections fail
- Manual firewall rule created but not working

**Solution:**
```bash
# Correct Configuration:
Destination: "WAN address" (not specific IP like 192.168.2.229)

# Incorrect (manual rule):
Destination: 192.168.2.229 → Will fail with DHCP changes

# Correct (automatic rule):
Destination: WAN address → Adapts to IP changes
```

#### Issue 3: Testing from Wrong IP

**Symptoms:**
- Connection to LAN IP instead of WAN IP
- Using internal IP from external network

**Solution:**
```bash
# Wrong (from external network):
http://192.168.10.1:8080 → LAN IP won't work

# Correct (from external network):
http://192.168.2.229:8080 → WAN IP with forwarded port
```

### Diagnostic Commands

```bash
# Check NAT status
Status → System Logs → Firewall

# Verify port forward rules
Firewall → NAT → Port Forward

# Test internal connectivity
ping 192.168.10.20

# Check WAN IP assignment
Status → Interfaces → WAN
```

## 📚 Implementation Phases

### Phase 1: NAT Assessment
- ✅ Review existing outbound NAT configuration
- ✅ Verify automatic NAT rule generation
- ✅ Confirm WAN interface IP assignment

### Phase 2: Port Forward Configuration
- ✅ Verify DVWA service accessibility (internal)
- ✅ Create port forward rule (WAN:8080 → 192.168.10.20:80)
- ✅ Configure filter rule association

### Phase 3: Firewall Rule Integration
- ✅ Enable automatic associated filter rule creation
- ✅ Remove incorrect manual rules
- ✅ Verify rule relationships

### Phase 4: Testing & Validation
- ✅ Test internal access (192.168.10.x → 192.168.10.20:80)
- ✅ Test external access (192.168.2.x → 192.168.2.229:8080)
- ✅ Verify security boundaries (Internet blocked)

## 🎓 Skills Demonstrated

### Technical Competencies

**Network Address Translation (NAT)**
- Port forwarding configuration and optimization
- Understanding NAT rule hierarchies and processing
- Filter rule association concepts
- Troubleshooting NAT connectivity issues

**Security Architecture**
- Controlled service exposure for security testing
- Network segmentation during service exposure
- Attack surface management via targeted port forwarding
- Risk assessment for external service accessibility

**Penetration Testing Infrastructure**
- Attack scenario setup and validation
- Controlled environment configuration
- Realistic attack simulation infrastructure
- External access for security testing

### Career Relevance

This project aligns with roles in:
- **Network Security Engineer**: NAT policy development, firewall optimization
- **Penetration Tester**: Attack simulation infrastructure setup
- **Security Analyst**: Service monitoring and traffic analysis
- **Security Architect**: Controlled testing environment design

## 🔄 Next Steps

### Immediate Actions
- [ ] Implement DNS filtering (Pi-hole - Project 3)
- [ ] Configure traffic monitoring (SPAN/TAP - Project 4)
- [ ] Deploy network security monitoring (Zeek - Project 5)

### Future Enhancements
- [ ] Additional service port forwarding as needed
- [ ] Enhanced logging for forwarded traffic
- [ ] Source IP restriction refinement
- [ ] 1:1 NAT for specific testing scenarios
- [ ] VPN integration with NAT for remote access

## 📖 Key Learnings

### Configuration Dependencies
- Port forward rules **require** associated filter rules for functionality
- Automatic rule association preferred over manual rule creation
- "WAN address" aliases provide better flexibility than hardcoded IPs
- Understanding rule relationships is critical for troubleshooting

### Security Considerations
- Controlled exposure preferable to Internet exposure for lab environments
- Network segmentation maintainable during service exposure
- Traffic logging essential for security monitoring
- Access source restriction improves security posture

### Troubleshooting Methodology
- Systematic verification of each configuration component
- Traffic flow analysis from source to destination
- Log analysis for identifying rule processing issues
- Step-by-step validation of rule relationships

## 📚 References

### Official Documentation
- [pfSense NAT Documentation](https://docs.netgate.com/pfsense/en/latest/nat/)
- [Port Forwarding Best Practices](https://docs.netgate.com/pfsense/en/latest/nat/port-forwards.html)
- [pfSense Firewall Rules Guide](https://docs.netgate.com/pfsense/en/latest/firewall/)

### Technical Standards
- [RFC 3022](https://tools.ietf.org/html/rfc3022) - Traditional IP Network Address Translator
- [RFC 2663](https://tools.ietf.org/html/rfc2663) - NAT Terminology and Considerations
- [NIST SP 800-41 Rev 1](https://csrc.nist.gov/publications/detail/sp/800-41/rev-1/final) - Guidelines on Firewalls and Firewall Policy

### Security Testing Resources
- [DVWA Documentation](http://www.dvwa.co.uk/) - Damn Vulnerable Web Application
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/) - Web Application Security Testing
- [PTES](http://www.pentest-standard.org/) - Penetration Testing Execution Standard

## 👤 Author

**Prageeth Panicker**

- GitHub: [@pragepani]((https://github.com/pragepani)
- LinkedIn: [LinkedIn](https://www.linkedin.com/in/prageeth-panicker)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- pfSense community for comprehensive NAT documentation
- Netgate for developing and maintaining pfSense
- DVWA project for providing security testing applications
- Cybersecurity home lab community for best practices

---

**Project Status**: ✅ Implemented and Validated  
**Last Updated**: October 24, 2025  
**Version**: 1.0

---

*Part of a comprehensive cybersecurity home lab project series. This project builds upon Project 1 (Multi-VLAN Deployment) and establishes the foundation for external attack simulation capabilities.*
