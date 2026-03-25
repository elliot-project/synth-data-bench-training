import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import importlib
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from megatron.energon import get_train_dataset, get_loader, WorkerConfig

def plot_token_map(matrix, colors, cat_map, output_path, title):
    """Renders a dense 2D pixel image of the token distributions."""
    fig, ax = plt.subplots(figsize=(14, max(4, matrix.shape[0] * 0.4)))
    cmap = ListedColormap(colors)
    
    ax.imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=len(colors)-1)
    
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels([f"Seq {i}" for i in range(matrix.shape[0])])
    ax.set_xlabel("Token Position")
    ax.set_title(title)
    
    legend_elements = [Patch(facecolor=colors[i], label=list(cat_map.keys())[i]) for i in range(len(colors))]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Image map saved to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Data-agnostic Energon Visualization.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to Energon dataset.")
    parser.add_argument("--encoder-module", type=str, default="task_encoders", help="Module containing the TaskEncoder.")
    parser.add_argument("--encoder-class", type=str, required=True, help="TaskEncoder class name.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Model ID for the encoder.")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--output", type=str, default="token_map.png", help="Output filename.")
    args = parser.parse_args()

    # Dynamic loading of the TaskEncoder
    module = importlib.import_module(args.encoder_module)
    encoder_cls = getattr(module, args.encoder_class)
    encoder = encoder_cls(model_id=args.model_id, max_length=args.max_length)

    print(f"Using {args.encoder_class} from {args.encoder_module}")
    print(f"Loading dataset from {args.dataset}...")
    
    dataset = get_train_dataset(
        args.dataset,
        worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0),
        task_encoder=encoder,
        batch_size=args.batch_size,
        shuffle_buffer_size=10,
        max_samples_per_sequence=None,
        packing_buffer_size=getattr(encoder, "packing_buffer_size", None)
    )
    loader = get_loader(dataset)

    print(f"🔄 Processing 1 batch...")
    batch = next(iter(loader))
    
    # Delegate categorization to the encoder (makes this file data-agnostic)
    matrix = encoder.categorize(
        batch['input_ids'], 
        batch['attention_mask'], 
        batch.get('cu_seqlens')
    )
    
    title = f"{args.encoder_class} | {args.model_id} | {args.dataset}"
    plot_token_map(matrix, encoder.COLORS, encoder.CAT_MAP, args.output, title)

if __name__ == "__main__":
    main()
