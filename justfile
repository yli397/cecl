# LMPO NCO Training Justfile
# Usage: just <command>

# Default variables
model_dir := "./checkpoints/Qwen3-1.7B/"
base_env := "JAX_PLATFORMS=cuda XLA_FLAGS=\"--xla_gpu_deterministic_ops=true\""

# Show available commands
default:
    @just --list

# Convert HuggingFace model to JAX format
convert-model hf_path="./checkpoints/hf/Qwen--Qwen3-1.7B/" jax_path="./checkpoints/Qwen3-1.7B/":
    python cecl/models/hf_to_jax.py --hf_dir {{hf_path}} --model_dir {{jax_path}}

# Test model sampling
test-sampling:
    python cecl/core/sampling.py --ckpt-dir {{model_dir}}

# CECL NCO prediction training (original 4-quarter)
train-nco name="NCO_H100" batch="24":
    {{base_env}} python cecl/core/grpo.py \
        --wandb_name {{name}} \
        --env_name nco \
        --model_dir {{model_dir}} \
        --groups_per_batch {{batch}} \
        --ppo_minibatch {{batch}} \
        --num_generation_tokens 1024 \
        --inference_batch_per_device 4 \
        --prompt_length 512

# Enhanced NCO training with temporal features (4-quarter)
train-nco-enhanced name="NCO_Enhanced" batch="16":
    {{base_env}} python cecl/core/grpo.py \
        --wandb_name {{name}} \
        --env_name nco_enhanced \
        --model_dir {{model_dir}} \
        --groups_per_batch {{batch}} \
        --ppo_minibatch {{batch}} \
        --num_generation_tokens 1536 \
        --inference_batch_per_device 3 \
        --prompt_length 768

# Single-quarter NCO prediction training (based on user feedback)
train-nco-single name="NCO_Single_Quarter" batch="32":
    {{base_env}} python cecl/core/grpo.py \
        --wandb_name {{name}} \
        --env_name nco_single \
        --model_dir {{model_dir}} \
        --groups_per_batch {{batch}} \
        --ppo_minibatch {{batch}} \
        --num_generation_tokens 512 \
        --inference_batch_per_device 8 \
        --prompt_length 512

# NCO ensemble training (combines multiple approaches)
train-nco-ensemble name="NCO_Ensemble" batch="12":
    {{base_env}} python cecl/core/grpo.py \
        --wandb_name {{name}}_GRPO \
        --env_name nco_enhanced \
        --model_dir {{model_dir}} \
        --groups_per_batch {{batch}} \
        --ppo_minibatch {{batch}} \
        --num_generation_tokens 1536 \
        --inference_batch_per_device 2 \
        --prompt_length 768

# Walk-forward validation for NCO
validate-nco checkpoint_dir start_date="2019Q1" end_date="2023Q4":
    {{base_env}} python cecl/core/eval.py \
        --env_name nco_enhanced \
        --model_dir {{checkpoint_dir}} \
        --walk_forward \
        --start_date {{start_date}} \
        --end_date {{end_date}}

# Evaluate trained NCO model
eval-nco env_name="nco_single" checkpoint_dir="{{model_dir}}":
    {{base_env}} python cecl/core/eval.py \
        --env_name {{env_name}} \
        --model_dir {{checkpoint_dir}}

# Monitor GPU usage
gpu-status:
    nvidia-smi

# Monitor training progress
monitor:
    htop -p $(pgrep -f "python.*grpo.py")

# Clean up checkpoints older than 7 days
clean-old-checkpoints:
    find ./checkpoints -name "step*" -type d -mtime +7 -exec rm -rf {} +

# Quick setup - install dependencies
setup:
    uv pip install jinja2
    @echo "Setup complete. Model conversion may be needed."
    @echo "Run: just convert-model"

# Resume training from checkpoint
resume-training checkpoint_dir name="Resume":
    {{base_env}} python cecl/core/grpo.py \
        --wandb_name {{name}} \
        --model_dir {{checkpoint_dir}} \
        --resume_training

# Debug NCO training with verbose output
debug-nco env="nco_single" name="Debug":
    JAX_PLATFORMS=cuda \
    JAX_TRACEBACK_FILTERING=off \
    XLA_FLAGS="--xla_gpu_deterministic_ops=true" \
    python cecl/core/grpo.py \
        --wandb_name {{name}} \
        --env_name {{env}} \
        --model_dir {{model_dir}} \
        --groups_per_batch 8 \
        --ppo_minibatch 8

# Analyze NCO training results
analyze-nco log_file="nco_predictions.json":
    python cecl/core/nco_analysis.py --log_file {{log_file}}