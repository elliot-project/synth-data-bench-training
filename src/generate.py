import os
import sys
import argparse
import uuid
import random
import io
import json
import tomllib
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
    num_images_per_sample = dataset_cfg.get("num_images", 1)
    num_turns = dataset_cfg.get("num_turns", 1)
    interleaved = dataset_cfg.get("interleaved", False)
    
    image_cfg = config.get("image", {})
    res_type = image_cfg.get("resolution_type", "fixed")
    
    os.makedirs(output_dir, exist_ok=True)
    shard_pattern = os.path.join(output_dir, "shard-%06d.tar")
    
    print(f"Generating {num_samples} samples to {output_dir}...")
    print(f"  Images: {num_images_per_sample}, Turns: {num_turns}, Interleaved: {interleaved}")
    
    with wds.ShardWriter(shard_pattern, maxsize=1e9, maxcount=1000) as sink:
        for i in range(num_samples):
            # Generate images
            images_data = []
            for _ in range(num_images_per_sample):
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
                images_data.append(img_byte_arr.getvalue())
            
            conversations = []
            images_placed = 0
            
            for turn in range(num_turns):
                # Human Turn
                human_text = LOREM_IPSUM[:random.randint(20, 100)]
                
                if interleaved:
                    if images_placed < num_images_per_sample:
                        human_text = "<image>\n" + human_text
                        images_placed += 1
                else:
                    if turn == 0:
                        human_text = "\n".join(["<image>"] * num_images_per_sample) + "\n" + human_text
                        images_placed = num_images_per_sample
                
                conversations.append({"from": "human", "value": human_text})
                
                # GPT Turn
                gpt_text = LOREM_IPSUM[:random.randint(50, 200)]
                conversations.append({"from": "gpt", "value": gpt_text})
            
            metadata = {
                "id": str(uuid.uuid4()),
                "conversations": conversations,
            }
            
            # Follow requested schema: Save every image in a separate key
            data = {
                "__key__": f"sample_{i:08d}",
                "json": metadata,
            }
            
            # jpg, jpg_1, jpg_2, ...
            for idx, img_bytes in enumerate(images_data):
                key = "jpg" if idx == 0 else f"jpg_{idx}"
                data[key] = img_bytes
            
            sink.write(data)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples")
            
    print(f"Finished writing WebDataset shards to {output_dir}")
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
