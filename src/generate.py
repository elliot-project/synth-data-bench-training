import os
import sys
import argparse
import random
import io
import json
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import torch
from torchvision.transforms.functional import to_pil_image
import webdataset as wds

LOREM_IPSUM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."

def generate_random_image(width, height):
    # Generate random RGB image
    tensor = torch.rand(3, height, width)
    return to_pil_image(tensor)

def generate_dataset(config_path):
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    
    dataset_cfg = config.get("dataset", {})
    num_samples = dataset_cfg.get("num_samples", 1000)
    output_dir = dataset_cfg.get("output_dir", "output_dataset")
    
    image_cfg = config.get("image", {})
    res_type = image_cfg.get("resolution_type", "fixed")
    
    os.makedirs(output_dir, exist_ok=True)
    shard_pattern = os.path.join(output_dir, "shard-%06d.tar")
    
    print(f"Generating {num_samples} samples to {output_dir}...")
    
    with wds.ShardWriter(shard_pattern, maxsize=1e9, maxcount=1000) as sink:
        for i in range(num_samples):
            if res_type == "fixed":
                w = image_cfg.get("width", 336)
                h = image_cfg.get("height", 336)
            elif res_type == "varying":
                min_res = image_cfg.get("min_res", 224)
                max_res = image_cfg.get("max_res", 1024)
                w = random.randint(min_res, max_res)
                h = random.randint(min_res, max_res)
            else:
                w, h = 336, 336
            
            img = generate_random_image(w, h)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            
            # Simulate visual-instruct data
            metadata = {
                "id": f"sample_{i:08d}",
                "conversations": [
                    {"from": "human", "value": f"<image>\n{LOREM_IPSUM[:random.randint(20, 100)]}?"},
                    {"from": "gpt", "value": LOREM_IPSUM[:random.randint(50, 200)]}
                ],
                "image_width": w,
                "image_height": h
            }
            
            sink.write({
                "__key__": f"sample_{i:08d}",
                "jpg": img_bytes,
                "json": metadata,
            })
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples")
            
    print(f"Finished writing WebDataset shards to {output_dir}")
    print(f"To prepare for Megatron-Energon, run:")
    print(f"  energon prepare {output_dir}")
    print()

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic WebDatasets.")
    parser.add_argument("configs", nargs="+", help="Paths to TOML configuration files.")
    args = parser.parse_args()
    
    for config_path in args.configs:
        print(f"Processing config: {config_path}")
        generate_dataset(config_path)

if __name__ == "__main__":
    main()