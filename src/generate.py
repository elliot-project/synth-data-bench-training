import os
import sys
import argparse
import uuid
import random
import math
import io
import json
import tomllib
import torch
from torchvision.transforms.functional import to_pil_image
import webdataset as wds

LOREM_IPSUM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
_LOREM_SOURCE = (LOREM_IPSUM + " ") * 210 # veryyy long sequence

def generate_random_image(width, height):
    # Generate random RGB image
    tensor = torch.rand(3, height, width)
    return to_pil_image(tensor)

def _sample_lognormal_resolution(image_cfg):
    """Sample (width, height) via a bivariate log-normal over size × aspect-ratio.

    Two independent normal draws are made:
      s  = exp(N(mu_log, sigma_log))  — geometric-mean side length
      ar = exp(N(0, ar_sigma))        — width / height ratio (centred on 1 = square)

    Width and height are then:
      W = s * sqrt(ar),  H = s / sqrt(ar)

    This produces the diagonal band (square-biased) seen in real instruct-dataset
    resolution heatmaps, while preserving the long tail toward high resolution.

    Config keys (all optional):
      mu_log    – mean of ln(geo-mean side),  default ln(500) ≈ 6.21
      sigma_log – std  of ln(geo-mean side),  default 0.55
      ar_sigma  – std  of ln(W/H),            default 0.35
                  (≈68 % of images between 0.70:1 and 1.43:1 aspect ratio)
      min_res   – hard lower clamp,           default 224
      max_res   – hard upper clamp,           default 2048
      round_to  – snap W and H to this grid,  default 0 (no snapping)
                  use 32 to match Qwen3-VL patch grid (16 px patch × merge-2)
    """
    mu = image_cfg.get("mu_log", math.log(500))
    sigma = image_cfg.get("sigma_log", 0.55)
    ar_sigma = image_cfg.get("ar_sigma", 0.35)
    min_res = image_cfg.get("min_res", 224)
    max_res = image_cfg.get("max_res", 2048)
    round_to = int(image_cfg.get("round_to", 0))

    geo_mean = math.exp(random.gauss(mu, sigma))
    ar = math.exp(random.gauss(0, ar_sigma))  # W / H

    w = int(round(geo_mean * math.sqrt(ar)))
    h = int(round(geo_mean / math.sqrt(ar)))

    def _clamp_snap(v):
        v = max(min_res, min(max_res, v))
        if round_to > 0:
            v = max(round_to, round(v / round_to) * round_to)
            v = min(max_res, v)
        return v

    return _clamp_snap(w), _clamp_snap(h)


def _sample_assistant_chars_per_turn(text_cfg, num_turns):
    """Sample per-turn assistant character count from a sample-level log-normal.

    Total assistant tokens for the sample are drawn once, then divided evenly
    across turns.  Sampling at the sample level (not per-turn) preserves the
    real distribution's median and mean exactly.

    Config keys (all optional):
      mu_log_assistant    – mean of ln(total tokens),  default ln(139) ≈ 4.934
      sigma_log_assistant – std  of ln(total tokens),  default 1.524
      chars_per_token     – conversion factor,          default 4.0
    """
    mu = text_cfg.get("mu_log_assistant", math.log(139))
    sigma = text_cfg.get("sigma_log_assistant", 1.524)
    chars_per_token = float(text_cfg.get("chars_per_token", 4.0))

    total_tokens = max(1, int(round(math.exp(random.gauss(mu, sigma)))))
    per_turn_tokens = max(1, total_tokens // max(1, num_turns))
    chars = int(per_turn_tokens * chars_per_token)
    return max(1, min(len(_LOREM_SOURCE), chars))


def generate_dataset(config_path):
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    
    dataset_cfg = config.get("dataset", {})
    num_samples = dataset_cfg.get("num_samples", 1000)
    output_dir = dataset_cfg.get("output_dir", "output_dataset")
    num_turns = dataset_cfg.get("num_turns", 1)
    interleaved = dataset_cfg.get("interleaved", False)

    _img_weights = dataset_cfg.get("num_images_weights", None)
    if _img_weights:
        _img_counts = list(range(1, len(_img_weights) + 1))
    else:
        _fixed_num_images = dataset_cfg.get("num_images", 1)

    image_cfg = config.get("image", {})
    res_type = image_cfg.get("resolution_type", "fixed")
    text_cfg = config.get("text", {})

    os.makedirs(output_dir, exist_ok=True)
    shard_pattern = os.path.join(output_dir, "shard-%06d.tar")

    if _img_weights:
        print(f"Generating {num_samples} samples to {output_dir}...")
        print(f"  Image count distribution: {dict(zip(_img_counts, _img_weights))}, Turns: {num_turns}, Interleaved: {interleaved}")
    else:
        print(f"Generating {num_samples} samples to {output_dir}...")
        print(f"  Images: {_fixed_num_images}, Turns: {num_turns}, Interleaved: {interleaved}")

    with wds.ShardWriter(shard_pattern, maxsize=1e9, maxcount=1000) as sink:
        for i in range(num_samples):
            if _img_weights:
                num_images_per_sample = random.choices(_img_counts, weights=_img_weights)[0]
            else:
                num_images_per_sample = _fixed_num_images

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
                elif res_type == "lognormal":
                    w, h = _sample_lognormal_resolution(image_cfg)
                else:
                    w, h = 336, 336
                
                img = generate_random_image(w, h)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                images_data.append(img_byte_arr.getvalue())
            
            conversations = []
            images_placed = 0
            # Sample total assistant tokens once per sample, reuse across turns
            asst_chars = _sample_assistant_chars_per_turn(text_cfg, num_turns)

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
                gpt_text = _LOREM_SOURCE[:asst_chars]
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
