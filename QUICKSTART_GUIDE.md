# Agentic-RIS Quick Start Guide

## Installation

```bash
# 1. Clone repository
cd agentic-ris-project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Setup API Keys

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API key
nano .env
# Add: CEREBRAS_API_KEY=your_key_here
```

## Quick Run

### 1. Test System Model

```bash
python src/core/system_model.py
```

### 2. Evaluate Baselines

```bash
python experiments/evaluate_all.py
```

This generates 3 plots in `results/plots/`:

- `sum_rate_vs_distance.png`
- `fairness_vs_distance.png`
- `aperture_influence.png`

### 3. Train DRL Baselines

```bash
# Train SAC for near-field
python experiments/train_drl_baselines.py --agent sac --regime near

# Train SAC for far-field
python experiments/train_drl_baselines.py --agent sac --regime far

# Train PPO
python experiments/train_drl_baselines.py --agent ppo --regime near
```

### 4. Train Agentic-RIS (Main Contribution)

```bash
python experiments/train_agentic.py
```

### 5. Run Complete Benchmark

```bash
python experiments/benchmark.py
```

## Project Structure
