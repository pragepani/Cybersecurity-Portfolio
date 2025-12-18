# Enterprise Security Architecture - Capstone Project

## Overview
A full-scale enterprise security architecture implementation for a mid-sized firm, designed, implemented, and validated in a virtualized lab environment (EVE-NG).

## Architecture Components
- **Network Security**: 15 VLAN segmentation with dual pfSense firewalls (80+ rules)
- **High Availability**: CARP for firewall HA, HSRP for gateway redundancy
- **Security Monitoring**: Wazuh EDR + Splunk SIEM integration
- **Vulnerability Management**: Nessus scanning and remediation tracking
- **SOC Operations**: Incident response workflows with documented playbooks

## Project Structure