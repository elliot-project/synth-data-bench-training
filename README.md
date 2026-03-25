# Synthetic Multimodal WebDatasets for Benchmarking

This project provides a synthetic data generation and packing pipeline for benchmarking large-scale training frameworks (VLM) using `megatron-energon` and `webdataset`.

## Project Structure
- `src/generate.py`: Main script for synthetic dataset generation.
- `src/task_encoders.py`: Contains `StandardVQATaskEncoder` and `PackingVQATaskEncoder` for data loading.
- `src/viz_synthetic.py`: Data-agnostic visualization tool for batch/token inspection.
- `configs/`: TOML configuration files for dataset generation parameters.
- `data/`: Location where generated datasets (WebDataset shards and Energon metadata) are stored.

## Usage Guide

### 1. Generating Synthetic Datasets
Generate shards with fixed or varying resolutions:
```bash
uv run python src/generate.py configs/dataset_standard.toml configs/dataset_varying.toml
```

### 2. Preparing for Energon
If you change the shards, you must re-index and prepare for Energon (required for the metadata and `.nv-meta` directory):
```bash
uv run energon prepare data/dataset_standard --non-interactive --split-ratio 0.8,0.1,0.1
uv run energon prepare data/dataset_varying --non-interactive --split-ratio 0.8,0.1,0.1
```

### 3. Visualizing Token Maps
Inspect how the `TaskEncoder` batches tokens and images. This works for both standard padding and optimized packing.

**Standard Padding:**
```bash
uv run python src/viz_synthetic.py \
    --dataset data/dataset_standard \
    --encoder-class StandardVQATaskEncoder \
    --output visualizations/standard_viz.png
```

**Optimized Packing:**
```bash
uv run python src/viz_synthetic.py \
    --dataset data/dataset_standard \
    --encoder-class PackingVQATaskEncoder \
    --output visualizations/packing_viz.png
```

## Features
- **Configurable Resolutions:** Standard (336x336) and varying (224-1024) image support.
- **Efficient Packing:** Custom `PackingVQATaskEncoder` that groups multiple samples into a single sequence, significantly reducing padding overhead and maximizing GPU throughput.
- **Diagnostic Tooling:** Integrated boundary markers (black pixels) in visualization to verify packing efficiency and FlashAttention-compatible `cu_seqlens`.
