# pfSense Multi-VLAN Deployment & Configuration

[![pfSense Version](https://img.shields.io/badge/pfSense-2.8.0-blue.svg)](https://www.pfsense.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-success.svg)]()

> A comprehensive implementation of pfSense firewall configuration with multi-VLAN network segmentation for a cybersecurity home lab environment.

## 🎯 Project Overview

This project documents the deployment and optimization of a pfSense firewall on MAC-MINI hardware, focusing on establishing baseline security policies for a segmented network environment. The implementation provides a foundation for advanced security monitoring, penetration testing, and network security analysis.

### Key Features

- **Network Segmentation**: Multi-VLAN architecture with dedicated Management, Internal, and User-LAN networks
- **Security Policies**: Documented firewall rules for traffic classification and monitoring
- **Attack Simulation**: Maintained penetration testing capabilities with proper attribution
- **Enhanced Logging**: Strategic logging implementation for traffic pattern analysis
- **Optimized Rules**: Elimination of redundant policies while maintaining operational flexibility

## 🏗️ Network Architecture

```
Internet
    │
    └─── WAN (em0) ─── pfSense (MAC-MINI) ─── LAN (em1) ─── Management Network (192.168.10.0/24)
                              │                                      │
                              │                                      ├─── Kali Attack Box (192.168.10.20)
                              │                                      ├─── DVWA Server
                              │                                      └─── Admin Workstations
                              │
                              └─── pfSense VM ─── Internal Network (192.168.30.0/24)
                                            └─── User-LAN (192.168.40.0/24)
```

### Network Segments

| Network | CIDR | Purpose | Gateway |
|---------|------|---------|---------|
| Management | 192.168.10.0/24 | Administrative access, security tools | 192.168.10.1 |
| Internal | 192.168.30.0/24 | Security platforms (OpenCTI, etc.) | 192.168.30.1 |
| User-LAN | 192.168.40.0/24 | User devices and testing | 192.168.40.1 |

## 🚀 Quick Start

### Prerequisites

- **Hardware**: MAC-MINI with dual network interfaces
- **Software**: pfSense 2.8.0 or later
- **Network**: Managed VLAN-capable switch
- **Access**: Administrative credentials for pfSense web interface

### Installation Steps

1. **Access pfSense Interface**
   ```bash
   https://192.168.10.1
   ```

2. **Review Current Configuration**
   - Navigate to Firewall → Rules
   - Document existing rules and interfaces
   - Verify WAN and LAN interface settings

3. **Implement Security Policies**
   - Create documented rules for administrative traffic
   - Configure logging for traffic analysis
   - Remove redundant firewall rules

4. **Validate Configuration**
   - Test connectivity across network segments
   - Verify logging functionality
   - Confirm attack simulation capabilities

📖 For detailed implementation steps, refer to the [Complete Implementation Guide](IMPLEMENTATION.md).

## 🔧 Configuration Details

### Firewall Rules (LAN Interface)

| Priority | Rule | Source | Destination | Purpose |
|----------|------|--------|-------------|---------|
| 1 | Anti-Lockout | Any | pfSense:443 | System protection |
| 2 | Kali Traffic Documentation | 192.168.10.20 | Any | Log penetration testing |
| 3 | Management Web Services | 192.168.10.0/24 | 192.168.30.0/24:80,443 | Admin access to tools |
| 4 | Default Allow (Logged) | 192.168.10.0/24 | Any | Fallback with visibility |
| 5 | Default Allow IPv6 | LAN IPv6 | Any | IPv6 support |

### Security Features

- ✅ RFC 1918 private network blocking (WAN)
- ✅ Bogon network blocking (WAN)
- ✅ Stateful packet filtering
- ✅ Enhanced traffic logging
- ✅ Anti-lockout protection

## 📊 Project Results

### Performance Metrics

- **Rule Optimization**: 17% reduction in total rules (6→5)
- **Redundancy Elimination**: 100% removal of duplicate functionality
- **Logging Coverage**: 100% visibility on significant traffic patterns
- **Configuration Time**: 2 hours (analysis + implementation + testing)

### Security Improvements

- Enhanced traffic classification for security analysis
- Documented attack simulation activities
- Improved operational visibility
- Maintained lab flexibility with controlled access

## 🧪 Testing & Validation

### Test Coverage

- ✅ Administrative access to pfSense web interface
- ✅ Cross-network connectivity (Management → Internal)
- ✅ Attack simulation tools (Kali, nmap, web testing)
- ✅ Logging verification and traffic pattern analysis
- ✅ Security tool integration (OpenCTI access)

### Validation Commands

```bash
# Test connectivity to internal services
curl -I http://192.168.30.60:8080

# Verify firewall rule processing
tcpdump -i em1 -n

# Check pfSense logs
tail -f /var/log/filter.log
```

## 🛠️ Troubleshooting

### Common Issues

**Issue**: Rules not processing as expected
- **Solution**: Verify rule order (top-to-bottom processing)
- Check for more specific rules above general rules
- Review state table for existing connections

**Issue**: Logging not functioning
- **Solution**: Verify log settings under Status → System Logs → Settings
- Enable logging checkbox on specific rules
- Check log retention policies

**Issue**: Connectivity loss after changes
- **Solution**: Anti-lockout rule should prevent total lockout
- Access pfSense console directly if needed
- Restore from configuration backup

## 📚 Documentation

- [Complete Implementation Guide](IMPLEMENTATION.md) - Comprehensive guide covering all aspects: implementation steps, troubleshooting, testing, validation, and results

## 🎓 Skills Demonstrated

### Technical Competencies

- **Firewall Administration**: pfSense configuration, rule optimization, traffic analysis
- **Network Security**: Segmentation design, access control, policy development
- **Security Monitoring**: Log analysis, traffic pattern recognition, SIEM preparation
- **Documentation**: Professional standards, change management, technical writing

### Career Relevance

This project aligns with roles in:
- Network Security Engineer
- Firewall Administrator
- Security Analyst
- SOC Analyst
- Network Administrator

## 🔄 Next Steps

### Immediate Enhancements
- [ ] Implement advanced NAT configuration (Project 2)
- [ ] Deploy Pi-hole DNS filtering (Project 3)
- [ ] Configure SPAN/TAP for traffic analysis (Project 4)
- [ ] Integrate Zeek network monitoring (Project 5)

### Future Development
- [ ] SIEM integration for automated analysis
- [ ] Advanced IDS/IPS configuration
- [ ] VPN setup for remote access
- [ ] High availability deployment

## 📖 References

### Official Documentation
- [pfSense Documentation](https://docs.netgate.com/pfsense/)
- [pfSense Firewall Rules Guide](https://docs.netgate.com/pfsense/en/latest/firewall/)

### Standards & Best Practices
- [NIST SP 800-41 Rev 1](https://csrc.nist.gov/publications/detail/sp/800-41/rev-1/final) - Guidelines on Firewalls and Firewall Policy
- [RFC 1918](https://tools.ietf.org/html/rfc1918) - Address Allocation for Private Internets
- [CIS Controls](https://www.cisecurity.org/controls) - Network Monitoring and Defense

## 👤 Author

**Prageeth Panicker**

- GitHub: [@yourusername]((https://github.com/pragepani)
- LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/prageeth-panicker)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- pfSense community for comprehensive documentation
- Netgate for developing and maintaining pfSense
- Cybersecurity home lab community for best practices and guidance

---

**Project Status**: ✅ Implemented and Validated  
**Last Updated**: October 24, 2025  
**Version**: 1.0

---

*Part of a comprehensive cybersecurity home lab project series focusing on network security, threat detection, and security operations.*
