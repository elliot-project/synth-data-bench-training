import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Professional style for academic/report plots
sns.set_theme(style="whitegrid", context="talk")

def load_data(json_path):
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def print_stats_table(df, title="PER-SAMPLE STATISTICS"):
    """
    Prints a formatted ASCII table with per-sample statistics.
    """
    metrics = {
        "Total Tokens": "total_tokens",
        "Image Tokens": "image_tokens",
        "Assistant Tokens": "assistant_tokens",
        "Num Images": "num_images"
    }
    
    print("\n" + "="*85)
    print(f" {title} (Based on {len(df)} analyzed samples)")
    print(f"{'METRIC':<20} | {'MEAN':<10} | {'STD':<10} | {'MEDIAN':<10} | {'MIN':<8} | {'MAX':<8} | {'P95':<8}")
    print("-" * 85)
    
    for label, col in metrics.items():
        if col not in df.columns:
            continue
            
        data = df[col]
        mean_val = data.mean()
        std_val = data.std()
        median_val = data.median()
        min_val = data.min()
        max_val = data.max()
        p95_val = data.quantile(0.95)
        
        print(f"{label:<20} | {mean_val:<10.2f} | {std_val:<10.2f} | {median_val:<10.2f} | {min_val:<8} | {max_val:<8} | {p95_val:<8.2f}")
    
    print("="*85 + "\n")

def print_extrapolated_table(df, scaling_factor, dataset_label="FULL DATASET"):
    """
    Prints a table with extrapolated totals for the full dataset.
    """
    metrics = {
        "Total Samples": None, 
        "Total Tokens": "total_tokens",
        "Image Tokens": "image_tokens",
        "Assistant Tokens": "assistant_tokens",
        "Num Images": "num_images"
    }

    print("\n" + "="*60)
    print(f" EXTRAPOLATED TOTALS FOR {dataset_label}")
    print(f" (Scaling Factor: {scaling_factor:.2f}x)")
    print("-" * 60)
    print(f"{'METRIC':<25} | {'ESTIMATED TOTAL':<25}")
    print("-" * 60)

    # Samples count
    est_samples = len(df) * scaling_factor
    print(f"{'Total Samples':<25} | {est_samples:,.0f}")

    for label, col in metrics.items():
        if col is None: continue
        if col not in df.columns: continue

        total_sum = df[col].sum()
        est_total = total_sum * scaling_factor
        print(f"{label:<25} | {est_total:,.0f}")
    
    print("="*60 + "\n")

def plot_sequence_distribution(df, output_dir):
    """
    Histogram + KDE of Total Token Lengths with percentile lines.
    """
    plt.figure(figsize=(12, 6))
    
    data = df['total_tokens']
    
    short_thresh = np.percentile(data, 33)
    long_thresh = np.percentile(data, 66)
    extreme_thresh = np.percentile(data, 95)

    sns.histplot(data, bins=60, kde=True, color="skyblue", line_kws={'linewidth': 2})
    
    plt.axvline(short_thresh, color='green', linestyle='--', alpha=0.8, label=f'Short (P33: {int(short_thresh)})')
    plt.axvline(long_thresh, color='orange', linestyle='--', alpha=0.8, label=f'Medium (P66: {int(long_thresh)})')
    plt.axvline(extreme_thresh, color='red', linestyle='--', alpha=0.8, label=f'Long (P95: {int(extreme_thresh)})')
    
    plt.title('Distribution of Total Sequence Lengths')
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "distribution_total_tokens.png")
    plt.close()

def plot_token_components(df, output_dir):
    """
    Boxen plot comparing Image vs Assistant tokens (Log Scale).
    """
    plt.figure(figsize=(10, 8))
    
    plot_df = df[['image_tokens', 'assistant_tokens']].melt(var_name='Type', value_name='Count')
    plot_df = plot_df[plot_df['Count'] > 0]
    
    sns.boxenplot(data=plot_df, x='Type', y='Count', palette="viridis")
    plt.yscale('log')
    
    plt.title('Token Composition (Log Scale)')
    plt.ylabel('Token Count')
    plt.xlabel('Token Type')
    
    plt.tight_layout()
    plt.savefig(output_dir / "token_components_comparison.png")
    plt.close()

def plot_resolution_heatmap(df, output_dir):
    """
    Heatmap of image resolutions (Width x Height) with labeled standards.
    """
    all_res = []
    for res_list in df['resolutions']:
        if res_list:
            all_res.extend(res_list)
            
    if not all_res:
        print("No resolution data found for heatmap.")
        return

    widths, heights = zip(*all_res)
    
    plt.figure(figsize=(12, 10))
    # Using log scale for counts to make sparse outliers visible
    plt.hexbin(widths, heights, gridsize=40, cmap='Blues', mincnt=1, bins='log')
    cb = plt.colorbar(label='Count (Log Scale)')
    
    plt.title('Image Resolution Heatmap')
    plt.xlabel('Width (px)')
    plt.ylabel('Height (px)')
    
    # --- Standard Resolutions & Labels ---
    standards = [
        (1024, 1024, "Square 1K"),
        (512, 512, "Square 512"),
        (336, 336, "ViT-L (336)"),
        (1920, 1080, "FHD (1080p)"),
        (1080, 1920, "Vertical FHD"),
        (1280, 720, "HD (720p)"),
        # A4 Resolutions (W x H)
        (595, 842, "A4 (72dpi)"),
        (794, 1123, "A4 (96dpi)"), 
        (1240, 1754, "A4 (150dpi)"),
        (2480, 3508, "A4 (300dpi)") 
    ]
    
    # Get plot limits to ensure labels don't fly off screen
    max_w = max(widths) * 1.1
    max_h = max(heights) * 1.1
    plt.xlim(0, max_w)
    plt.ylim(0, max_h)

    # Add markers and text
    for w, h, label in standards:
        # Only plot if within reasonable range of data to avoid zooming out too far
        if w < max_w and h < max_h:
            plt.plot(w, h, 'rx', markersize=8, markeredgewidth=2)
            plt.text(w + (max_w * 0.02), h + (max_h * 0.01), label, 
                     color='red', fontsize=10, weight='bold',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    plt.tight_layout()
    plt.savefig(output_dir / "resolution_heatmap.png")
    plt.close()

def plot_images_per_sample(df, output_dir):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=df['num_images'], palette="magma")
    
    plt.title('Number of Images per Sample')
    plt.xlabel('Image Count')
    plt.ylabel('Samples')
    
    plt.tight_layout()
    plt.savefig(output_dir / "images_per_sample_distribution.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, default="extrapolated_stats_v2.json", help="Input JSON file")
    parser.add_argument("--output_dir", type=str, default="./plots", help="Directory to save plots")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Modified data loading to get full dict
    full_data = load_data(args.json_file)
    raw_list = full_data.get("raw_samples", [])
    
    # Extract metadata for extrapolation
    meta = full_data.get("meta", {})
    scaling_factor = meta.get("scaling_factor", 1.0)
    
    if not raw_list:
        print("Error: No 'raw_samples' found.")
        return
        
    df = pd.DataFrame(raw_list)
    
    # Filter for Image-Only samples
    df_img = df[df['has_image'] == True]
    
    print(f"\nLoaded {len(df)} total samples.")
    print(f"Analyzing {len(df_img)} samples containing images.")

    # Calculate and print empty sample stats
    num_no_image = len(df) - len(df_img)
    pct_no_image = (num_no_image / len(df)) * 100 if len(df) > 0 else 0
    print(f"Samples WITHOUT images: {num_no_image} ({pct_no_image:.2f}%)")

    # 1. Print Stats Tables (Per Sample)
    print_stats_table(df, title="PER-SAMPLE STATS (ENTIRE DATASET)")
    print_stats_table(df_img, title="PER-SAMPLE STATS (IMAGE-ONLY SAMPLES)")
    
    # 2. Print Extrapolated Totals Table (Full Dataset vs Image-Only)
    print_extrapolated_table(df, scaling_factor, dataset_label="ENTIRE DATASET (Including Text-Only)")
    print_extrapolated_table(df_img, scaling_factor, dataset_label="IMAGE-ONLY SAMPLES (VLM Training Data)")

    # 3. Generate Plots (Focusing on Image Data as per VLM context)
    print("Generating plots (using Image-Only samples)...")
    plot_sequence_distribution(df_img, out_path)
    plot_token_components(df_img, out_path)
    plot_resolution_heatmap(df_img, out_path)
    plot_images_per_sample(df_img, out_path)
    
    print(f"Done! Plots saved to {out_path.resolve()}")

if __name__ == "__main__":
    main()