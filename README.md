# Agentic-RIS: LLM-Driven XL-RIS Optimization Framework

A novel framework combining Large Language Models (LLMs) with Deep Reinforcement Learning (DRL) for optimizing Extra-Large Reconfigurable Intelligent Surfaces (XL-RIS) in mixed near-field and far-field wireless communication scenarios.

## 🎯 Key Features

- **Hybrid LLM-DRL Architecture**: Combines the reasoning capabilities of LLMs with the optimization power of DRL
- **Physics-Informed RAG**: Retrieval-Augmented Generation database with electromagnetic (EM) physics principles
- **Adaptive Regime Selection**: Automatically switches between near-field beamfocusing and far-field beamsteering
- **Depth-Division Multiplexing (DDM)**: Exploits spherical wavefront model for spatial separation
- **Dynamic Aperture Tapering**: LLM-guided aperture size selection for interference mitigation

## 📊 System Specifications

- **Base Station**: M=16 antennas (Uniform Linear Array)
- **XL-RIS**: N=4,096 elements (64×64 Uniform Planar Array)
- **Frequency**: 28 GHz (mmWave)
- **Channel Model**: Spherical Wavefront Model (SWM) for all links

## 🏗️ Project Structure

```
wcl-code/
├── src/
│   ├── core/                    # System model & channel computations
│   ├── agents/                  # DRL agents (SAC, PPO)
│   ├── agentic/                 # LLM & RAG components
│   ├── environment/             # Gym environment
│   └── utils/                   # Visualization & config
├── experiments/                 # Training & evaluation scripts
├── configs/                     # YAML configurations
├── data/
│   ├── rag_knowledge/          # Physics rules database
│   └── trained_models/         # Saved models
└── results/plots/              # Generated figures
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sameeramjad07/wcl-code.git
cd wcl-code

# Create virtual environment using venv or using Conda mini
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Configuration

1. Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

2. Edit `.env`:

```
CEREBRAS_API_KEY=your_cerebras_api_key_here
```

### Running Experiments

```bash
# Train DRL baselines
python experiments/train_drl_baselines.py --config configs/drl_config.yaml

# Train Agentic-RIS
python experiments/train_agentic.py --config configs/agentic_config.yaml

# Evaluate all methods and generate plots
python experiments/evaluate_all.py --config configs/system_config.yaml

# Run comprehensive benchmark
python experiments/benchmark.py
```

## 📈 Key Results

The framework generates four critical plots:

1. **Sum-Rate vs. User Distance**: Shows performance across mixed-field transition
2. **Jain's Fairness Index**: Demonstrates balanced user rate allocation
3. **Convergence Comparison**: Agentic-SAC vs. Blind-SAC training speed
4. **Aperture Size Influence**: SINR variation with active RIS elements

## 🧠 Methodology

### 1. LLM-Driven Strategy Selection

The LLM analyzes:

- User positions (distance, angle, height)
- Retrieved physics principles from RAG
- Current system state

And outputs:

- Optimal RIS aperture mask
- Reward function parameters (α, β, γ)
- Recommended beamforming strategy

### 2. DRL Fine-Tuning

SAC agent optimizes:

- **State**: [r_A, r_B, θ_A, θ_B, strategy_ID]
- **Action**: Phase shifts for active elements + power allocation
- **Reward**: R = α·log(1+SINR_A) + β·log(1+SINR_B) - γ·I_leakage

### 3. Key Innovations

- **Gradient Masking**: Only active RIS elements are updated during training
- **Warm Start**: LLM provides physics-informed initialization
- **Online Fine-Tuning**: 50-100 SAC steps (10-20ms) vs. full retraining

## 📊 Baselines

1. **Always Near-Field**: Fixed beamfocusing for all users
2. **Always Far-Field**: Fixed beamsteering for all users
3. **Threshold-Based**: Switches at d_FF boundary
4. **Exhaustive Search**: Quantized brute-force optimization

## 🔬 Technical Details

### Spherical Wavefront Model (SWM)

The channel from RIS element n to user k:

```
h_n,k = (λ / 4πr_n,k) · exp(-j2πr_n,k / λ)
```

where r_n,k is the Euclidean distance between element n and user k.

### Cascaded Channel

```
H_k^H = h_{R,k}^H · Φ · G
```

where:

- h\_{R,k}: RIS-to-user channel
- Φ: Diagonal matrix of RIS phase shifts
- G: BS-to-RIS channel matrix

## 📝 Citation

If you use this code in your research, please cite using the cite this repository option of Github on this repo.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaborations, please contact: proton7ve@gmail.com

## 🙏 Acknowledgments

- UC Berkeley for SAC algorithm
- Cerebras for high-speed LLM inference
