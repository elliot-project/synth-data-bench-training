# Synthetic Multimodal WebDatasets for Benchmarking

This repository provides a pipeline for generating synthetic multimodal datasets and packing them using `megatron-energon` and `webdataset`. It is designed as a starting point for VLM (Vision-Language Model) training frameworks, demonstrating how to handle interleaved images, multi-turn conversations, and efficient data packing.

> Contact: `tockier@cvc.uab.cat` (Computer Vision Center)

## Features
- **Synthetic Generation**: Generate massive datasets with randomized text (Lorem Ipsum) and random images (Gaussian noise).
- **Multimodal Support**: Support for captioning (1 image), VQA (multiple turns), and interleaved data (multiple images per sample).
- **Megatron-Energon Integration**: Ready-to-use `TaskEncoder` and `Cookers` for `megatron-energon`.
- **Data Packing**: Demonstrates how to pack multiple variable-length samples into a single fixed-length sequence using `cu_seqlens`.
- **Diagnostic Tools**: High-fidelity visualization of token distributions and decoded text in packed batches.

## Project Structure
- `src/generate.py`: Main script for synthetic dataset generation.
- `src/task_encoders.py`: Contains `TaskEncoder` implementations and `Cookers`.
- `src/viz_synthetic.py`: Visualizes token distributions (image vs. text vs. padding).
- `src/viz_text.py`: Decodes and prints the text within packed batches.
- `configs/`: TOML configuration files for various dataset types (Captioning, VQA, Interleaved).
- `ENERGON_DOCS.md`: Detailed documentation on Energon integration.

## Installation
This project uses `uv` for dependency management. To set up the environment:
```bash
uv sync
```

## Workflow

### 1. Generate Synthetic Datasets
Use the provided configurations or create your own to generate `WebDataset` shards:
```bash
# Generate a simple VQA dataset
uv run python src/generate.py configs/vqa.toml

# Generate an interleaved dataset with multiple images
uv run python src/generate.py configs/interleaved.toml
```

### 2. Prepare for Energon
Before using the dataset with Energon, you must prepare the metadata:
```bash
uv run energon prepare data/vqa --non-interactive --split-ratio 1.0,0,0 --sample-type CrudeSample --force-overwrite
```
*Note: We use `CrudeSample` to keep the raw data accessible to our custom Cookers.*

### 3. Visualization and Inspection
Verify that the data is being loaded and packed correctly.

#### Token Distribution Map
Visualize how User Text, Assistant Text, Images, and Padding are distributed in a batch:
```bash
uv run python src/viz_synthetic.py \
    --dataset data/vqa \
    --encoder-class DataPackingEncoder \
    --output visualizations/vqa_tokens.png
```

#### Text Inspector
Decode the actual text being fed to the model:
```bash
uv run python src/viz_text.py \
    --dataset data/vqa \
    --encoder-class DataPackingEncoder
```

## Examples

Captioning dataset with small images:
<img width="4174" height="1166" alt="captioning" src="https://github.com/user-attachments/assets/4cd2c5ad-139c-42e2-bcd2-31901e0ad802" />

VQA dataset with multiple user-assistant turns:
<img width="4174" height="1166" alt="vqa" src="https://github.com/user-attachments/assets/7d9a12ed-20de-45c8-b092-c4a0bb260b86" />

Interleaved dataset with multiple images in the same sample:
<img width="4174" height="1166" alt="interleaved" src="https://github.com/user-attachments/assets/ea5fbf3b-8e54-478f-9feb-b6902603163f" />

## Contributing
For more details on how to extend the `TaskEncoder` or add new `Cookers`, please refer to [ENERGON_DOCS.md](ENERGON_DOCS.md).

**Feel free to create any issues or PRs.**
