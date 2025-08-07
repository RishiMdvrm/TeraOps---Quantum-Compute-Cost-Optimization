# TeraOps — Quantum Compute Cost Optimization 🚀

**TeraOps** is a smart quantum compute optimizer that minimizes compute costs while maintaining performance. It simulates quantum workload allocation strategies and uses AI-based forecasting and reinforcement learning to optimize QPU (Quantum Processing Unit) resource usage over time.

> 📅 **Project for CS 532: Spring 2025**  
> 👨‍🏫 Instructor: Prof. Petrov  
> 🧠 Advisor: Manoj Rana  
> 👥 Team D

---

## 🌍 Context: Welcome to 2049

Quantum computing has fully replaced classical computing. You're scaling your app using quantum service providers—but your first monthly bill is $2 million. Confused by terms like *Atom, Photon, Spin, lease cost, execution cost*?

Enter **TeraOps** — your cost-aware quantum optimization assistant. It lets you build confidently without blowing your budget.

---

## 🧠 Project Objectives

- Simulate quantum resource usage over 180 days.
- Analyze different workload allocation strategies.
- Optimize costs via ML and RL-based methods.
- Forecast workload trends to guide compute allocation.
- Build a user-friendly dashboard and LLM-powered assistant for compute insights.

---

## 📦 Dataset Simulation

Simulated daily quantum workloads and QPU leasing behavior:

- **Workload Volume**: 1M–50M workloads/day
- **QPU Leases**: 1K–10K blocks/day
- **Timeline**: 180 simulated days
- **Data Format**: Optimized Parquet files
  - `blocks.parquet`
  - `workloads_daily.parquet`

### 💸 Cost Model

| Category           | Cost                     |
|--------------------|--------------------------|
| Lease              | $3/hr per block          |
| Acquisition        | $0.20 per block          |
| Trigger            | $0.01 per workload       |
| Execution          | $0.01–$0.20 per workload |

---

## ⚙️ Optimization Techniques

### 1. **Round Robin**
- Simple rotation logic
- No demand awareness
- Baseline model with high cost, low efficiency

### 2. **One-Shot Break Even**
- Tag blocks (Atom, Photon, Spin) **once** at lease time
- Based on average historical jobs/day:
  - Atom: > 900 jobs/day
  - Photon: 176–900 jobs/day
  - Spin: < 176 jobs/day

### 3. **EWMA Retagging**
- Uses **Exponential Weighted Moving Average** to track trends
- Reassigns tags dynamically to match workload shifts

### 4. **Rolling-7 Retagging**
- Computes 7-day sliding average for each block
- Retags based on recent workload trends

### 5. **Reinforcement Learning**
- Deep Q-Network (DQN) agent
- Trained on 180 days of data
- Learns optimal retagging policy to minimize total cost
- State: `[Days Left, 7-Day Avg Jobs, Current Tag, Today’s Jobs]`
- Reward: `-1 * total daily cost`

---

## 🔮 Forecasting

### LSTM-based Prediction
- Input: Past 14 days of workload data (per tag)
- Output: Next 7-day forecast
- Architecture: 2-layer LSTM with 64 hidden units
- Trained for 20 epochs to minimize MSE

Forecasts proactively guide retagging and leasing.

---

## 🤖 Agent & Assistant

### NLP Assistant Capabilities:
- Built on **GPT-4** via **LangChain**
- Answers natural language questions about:
  - QPU usage
  - Optimization strategies
  - Cost analysis
- Uses:
  - HuggingFace Embeddings + FAISS for vector search
  - SentenceTransformers for NLP
  - PyDantic for type validation
  - Plotly for visualizations
  - Scikit-learn (Linear Regression, Random Forest)

---

## 🖥️ Tech Stack

### Backend
- `FastAPI` (RESTful API)
- `PyArrow`, `Parquet` (high-volume data handling)
- `stable-baselines3` (Reinforcement Learning)
- `PyTorch` (LSTM training)

### Frontend
- `React` with `React Router`
- `Tailwind CSS` (styling)
- `Recharts` (dashboard visualization)

### Integration
- End-to-end orchestration from simulation → optimization → LLM assistant → dashboard

---

## 🚀 Future Scope
- **Multi-objective RL**: Balance cost, latency, and energy use
- **Cloud Integration**: Support AWS, Azure, GCP data pipelines
- **Scheduling**: Let users schedule assistant tasks for automation
- **Transformer Forecasting**: Replace LSTM with scalable transformers
- **Security & Privacy**: Enforce guardrails around assistant outputs

## 📜 License

This project is licensed under the [MIT License](./LICENSE).  
Copyright © 2025 **Rishi Madhavaram**

You are free to use, modify, and distribute this project as long as the original license and copyright notice are retained.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Acknowledgements
- UIC CS 532 (Spring 2025)
- Prof. Petrov, Plamen (UIC), Advisor Rana, Manoj
- Stable-Baselines3, PyTorch, HuggingFace, LangChain

## 💬 Questions?
Open an issue or ping us in the Discussions tab. Happy optimizing!
