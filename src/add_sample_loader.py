import os
import argparse
import sys

LOADER_TEMPLATE = """import torch

def sample_loader(raw: dict) -> dict:
    sequence = []
    
    # Gather all images available in the raw dict
    imgs = [raw.get("jpg")]
    idx = 1
    while f"jpg_{idx}" in raw:
        imgs.append(raw.get(f"jpg_{idx}"))
        idx += 1
    
    img_ptr = 0
    
    metadata = raw.get("json", {})
    if isinstance(metadata, list):
        conversations = metadata
    else:
        conversations = metadata.get("conversations") or metadata.get("messages") or []

    for turn in conversations:
        role = turn.get("from") or turn.get("role", "unknown")
        text = turn.get("value") or turn.get("content", "")
        
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant"):
            role = "assistant"
        
        parts = text.split("<image>")
        for i, part in enumerate(parts):
            if part:
                sequence.append({"role": role, "text": part})
            
            if i < len(parts) - 1:
                if img_ptr < len(imgs) and imgs[img_ptr] is not None:
                    sequence.append(imgs[img_ptr])
                    img_ptr += 1

    return dict(
        __key__=raw.get("__key__"),
        sequence=sequence, 
    )

def part_filter(part: str) -> bool:
    return part == 'json' or part.startswith('jpg')
"""

def main():
    parser = argparse.ArgumentParser(description="Add standard interleaved sample loader to an Energon dataset.")
    parser.add_argument("dataset_path", help="Path to the dataset directory (containing .nv-meta).")
    args = parser.parse_args()

    nv_meta_dir = os.path.join(args.dataset_path, ".nv-meta")
    loader_path = os.path.join(nv_meta_dir, "sample_loader.py")

    if not os.path.isdir(nv_meta_dir):
        print(f"Error: .nv-meta directory not found in {args.dataset_path}")
        sys.exit(1)

    print(f"Writing sample loader to {loader_path}...")
    with open(loader_path, "w") as f:
        f.write(LOADER_TEMPLATE)
    
    print("Successfully updated sample_loader.py")

if __name__ == "__main__":
    main()
