# System Setup Scripts

This directory contains scripts and systemd service files for setting up the WF4NYM capture environment.

## Overview

These scripts configure a system with network namespaces, Nym components, and traffic capture tools for website fingerprinting experiments on both proxy and network requester side.

## Files

### Setup Scripts

- **`setup_system.sh`** - Main system setup script that installs all dependencies and configures the environment

## Installation


### Steps

1. **Run the main setup script:**
   ```bash
   sudo ./setup_system.sh
   ```

2. **Configure Nym components:**
   
   Before enabling the services, update the `Environment` variable in the service files:
   
   Edit `/etc/systemd/system/wfp-proxy1.service`:
   ```bash
   Environment=NYM_PATH=/path/to/your/nym
   ```
   
   Edit `/etc/systemd/system/wfp-requester1.service`:
   ```bash
   Environment=NYM_PATH=/path/to/your/nym
   ```

3. **Initialize Nym components:**
    
    On the network requester side:
   ```bash
   cd ~/nym
   ./target/release/nym-network-requester init --id nym-requester

   ./target/release/nym-socks5-client init --id proxy --provider nym-requester
   ```

       On the proxy side:
   ```bash
   cd ~/nym

   ./target/release/nym-socks5-client init --id proxy --provider *REQUESTER ID*
   ```