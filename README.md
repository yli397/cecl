# CECL - NCO Prediction with GRPO-Enhanced LLM

## Overview

CECL (Current Expected Credit Loss) is a machine learning framework that leverages GRPO (Group Relative Policy Optimization) enhanced Large Language Models for NCO (Net Charge-Off) prediction. The system uses 12 quarters of historical financial data to predict the next quarter's net charge-offs, combining state-of-the-art LLM capabilities with financial time-series analysis for accurate credit loss forecasting.

## Key Features

- **GRPO-Enhanced Training**: Implements Group Relative Policy Optimization for improved model performance
- **Time-Series Forecasting**: Uses 12 quarters of historical data to predict the next quarter's NCO
- **Walk-Forward Validation**: Built-in time-series validation for financial data
- **GPU Optimization**: JAX-based implementation with CUDA acceleration
- **Flexible Architecture**: Supports multiple model configurations including Qwen3-1.7B

## Prerequisites

- Python 3.12+
- CUDA-enabled GPU (recommended: H100, A100, or V100)
- JAX with CUDA support
- UV package manager (for dependency management)

## Installation

### 1. Clone the Repository
```bash
git clone [repository-url]
cd cecl
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
uv pip install jinja2
# Install additional requirements as needed
```

### 3. Download Pre-trained Models
Place the Qwen3-1.7B model in the `checkpoints/` directory:
```bash
mkdir -p checkpoints/Qwen3-1.7B/
# Download model files to this directory
```

### 4. Convert HuggingFace Models to JAX Format (if needed)
```bash
just convert-model hf_path="./checkpoints/hf/Qwen--Qwen3-1.7B/" jax_path="./checkpoints/Qwen3-1.7B/"
```

## Quick Start

### Basic Training
Run NCO prediction training with default settings (uses 12 quarters of historical data):
```bash
just train-nco
```

### Custom Training Configuration
```bash
# Train with custom batch size and experiment name
just train-nco name="MyExperiment" batch="16"

# Enhanced training with temporal features and extended context
just train-nco-enhanced name="Enhanced_Model" batch="16"

# Single-quarter focused training (still uses 12Q history, optimized for single Q output)
just train-nco-single name="SingleQ_Test" batch="32"
```

## Usage Guide

### 1. Training Models

#### Standard NCO Training (12-Quarter Historical Data)
Uses 12 quarters of historical data to predict the next quarter's NCO:
```bash
JAX_PLATFORMS=cuda XLA_FLAGS="--xla_gpu_deterministic_ops=true" \
python cecl/core/grpo.py \
    --wandb_name NCO_Training \
    --env_name nco \
    --model_dir ./checkpoints/Qwen3-1.7B/ \
    --groups_per_batch 24 \
    --ppo_minibatch 24 \
    --num_generation_tokens 1024 \
    --inference_batch_per_device 4 \
    --prompt_length 512
```

#### Alternative Training Configurations
```bash
# Standard configuration with custom batch size
just train-nco name="Q1_2024_Prediction" batch="32"

# Enhanced training with extended context
just train-nco-enhanced name="Enhanced_Model" batch="16"
```

### 2. Model Evaluation

#### Basic Evaluation
```bash
just eval-nco env_name="nco_single" checkpoint_dir="./checkpoints/Qwen3-1.7B/"
```

#### Walk-Forward Validation
```bash
just validate-nco checkpoint_dir="./checkpoints/trained_model/" start_date="2019Q1" end_date="2023Q4"
```

### 3. Monitoring and Debugging

#### Monitor GPU Usage
```bash
just gpu-status
```

#### Debug Training Issues
```bash
just debug-nco env="nco_single" name="Debug_Run"
```

#### Monitor Training Progress
```bash
just monitor
```

## Advanced Features

### Resume Training from Checkpoint
```bash
just resume-training checkpoint_dir="./checkpoints/step_1000/" name="Resume_Training"
```

### Ensemble Training
Combines multiple approaches for improved accuracy:
```bash
just train-nco-ensemble name="Ensemble_Model" batch="12"
```

### Custom Environment Variables
```bash
export JAX_PLATFORMS=cuda
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Project Structure

```
cecl/
├── cecl/
│   ├── core/           # Core training and evaluation modules
│   │   ├── grpo.py     # GRPO implementation
│   │   ├── eval.py     # Evaluation scripts
│   │   └── sampling.py # Model sampling utilities
│   ├── models/         # Model architectures and conversions
│   │   └── hf_to_jax.py # HuggingFace to JAX converter
│   ├── envs/           # Environment configurations
│   └── utils/          # Utility functions
├── checkpoints/        # Model checkpoints
├── data/              # Training and validation data
├── results/           # Training results and logs
└── justfile          # Task automation commands
```

## Data Preparation

### ARCH Data Processing
For ARCH model comparisons:
```bash
python prepare_arch_data.py
```

### Visualization and Analysis
```bash
# Generate comparison charts
python create_comparison_png.py

# Analyze directional accuracy
python directional_accuracy_visualization.py

# Create professional reports
python professional_direction_chart.py
```

## Performance Optimization

### Batch Size Tuning
- H100 GPU: Use `batch="24"` or higher
- A100 GPU: Use `batch="16"`
- V100 GPU: Use `batch="8"`

### Memory Management
```bash
# Reduce memory fragmentation
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Enable JAX cache
export JAX_CACHE_DIR=/tmp/jax_cache
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `groups_per_batch` and `ppo_minibatch`
   - Decrease `inference_batch_per_device`

2. **JAX Platform Not Found**
   ```bash
   export JAX_PLATFORMS=cuda  # or cpu for CPU-only
   ```

3. **Model Loading Errors**
   - Ensure model files are in correct JAX format
   - Run conversion script if using HuggingFace models

### Cleanup Old Checkpoints
```bash
just clean-old-checkpoints
```

## Results and Logging

Training logs are saved to:
- WandB: Online tracking (if configured)
- Local logs: `nco_training.log`, `nco_training_new.log`
- Results directory: `./results/`

## Citation

If you use this framework in your research, please cite:
```bibtex
@software{cecl2025,
  title={CECL: NCO Prediction with GRPO-Enhanced LLM},
  author={Your Team},
  year={2025}
}
```

## Support

For issues and questions:
- Check existing issues in the repository
- Contact the maintainer: yli269
- Review documentation in `docs/` directory

## License

[Specify your license here]

## Acknowledgments

This project uses:
- Qwen3 language models
- JAX framework for accelerated computing
- GRPO optimization techniques