# Synthetic Multimodal WebDatasets for Benchmarking

This project provides a synthetic data generation and packing pipeline for benchmarking large-scale training frameworks (VLM) using `megatron-energon` and `webdataset`.

## Project Structure
- `src/generate.py`: Main script for synthetic dataset generation.
- `src/task_encoders.py`: Contains the `MultimodalTaskEncoder` which supports `InterleavedSample` objects.
- `src/add_sample_loader.py`: Utility to inject the interleaved sample loader into Energon metadata.
- `src/viz_synthetic.py`: Data-agnostic visualization tool for batch/token inspection.
- `configs/`: TOML configuration files for dataset generation.
- `data/`: Storage for generated datasets.

## Workflow (Transparency-Focused)

To ensure full control and transparency over the dataset preparation process, we follow a 3-step manual workflow instead of using opaque wrappers around the Energon CLI.

### 1. Generate Synthetic Datasets
Generate shards with multiturn conversations and interleaved images using your chosen configuration:
```bash
uv run python src/generate.py configs/dataset_interleaved.toml
```

### 2. Prepare for Energon
Use the standard `energon prepare` command. **Crucially**, specify `--sample-type InterleavedSample` to generate the necessary metadata stubs.
```bash
uv run energon prepare data/dataset_interleaved --non-interactive --split-ratio 1.0,0,0 --sample-type InterleavedSample --force-overwrite
```

### 3. Add the Sample Loader
Energon generates a stub `sample_loader.py` in the `.nv-meta` directory. Use our utility script to populate it with the logic required to handle the interleaved synthetic data:
```bash
uv run python src/add_sample_loader.py data/dataset_interleaved
```

## Visualization
After preparation, you can visualize the batches to verify packing efficiency and multimodal token distribution:
```bash
export PYTHONPATH=$PYTHONPATH:.
uv run python src/viz_synthetic.py \
    --dataset data/dataset_interleaved \
    --encoder-class MultimodalTaskEncoder \
    --output visualizations/interleaved_viz.png
```

## Features
- **Interleaved Multimodal Support**: Generates multiturn conversations with images placed at arbitrary points.
- **Enforced InterleavedSample**: All datasets are configured to use the generic `InterleavedSample` type, maximizing compatibility with VLM training frameworks.
- **Efficient Packing**: `MultimodalTaskEncoder` handles the packing of complex, variable-length interleaved sequences.
- **Diagnostic Tooling**: High-fidelity token maps with sequence boundary markers.
