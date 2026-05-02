# 64-bit Federated NIDPS
A decentralized Network Intrusion Detection System (NIDPS) using **Federated Learning** and **64-bit precision** MobileViT.

## 🚀 Overview
This project demonstrates a privacy-preserving AI pipeline where a central **Kali Linux** server aggregates mathematical updates from a **Windows/WSL** client.
- **Architecture:** MobileViT-XXS (Vision Transformer)
- **Precision:** Float64 (Double Precision) for high-fidelity gradient updates.
- **Frameworks:** Flower (FL), PyTorch, Timm.

## 🛠️ Setup
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Start the server on Kali: `python src/server.py`
4. Start the client on Windows: `python src/client.py`