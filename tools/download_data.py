#!/usr/bin/env python3
"""
Download sample datasets in LLaMA-Factory format for Gemma 3 VLM training
"""
import os
import json
from pathlib import Path
from tqdm import tqdm


def download_naruto_for_llamafactory(output_dir: str, num_samples: int = 500):
    """Download Naruto captions in LLaMA-Factory format"""
    from datasets import load_dataset
    
    print("Downloading Naruto-BLIP-Captions for LLaMA-Factory...")
    
    dataset = load_dataset("lambdalabs/naruto-blip-captions", split="train")
    
    if num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
    
    # Create output directories
    output_path = Path(output_dir)
    images_path = output_path / "images"
    output_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(exist_ok=True)
    
    # Convert to LLaMA-Factory format
    converted_data = []
    
    for idx, item in enumerate(tqdm(dataset, desc="Processing")):
        try:
            image = item["image"]
            caption = item["text"]
            
            image_filename = f"naruto_{idx:06d}.png"
            image_path = images_path / image_filename
            image.save(image_path)
            
            # LLaMA-Factory format
            converted_data.append({
                "messages": [
                    {"role": "user", "content": "<image>Describe this image in detail."},
                    {"role": "assistant", "content": caption}
                ],
                "images": [f"images/{image_filename}"]
            })
        except Exception as e:
            print(f"Warning: Failed item {idx}: {e}")
            continue
    
    # Save JSON
    with open(output_path / "train.json", "w", encoding="utf-8") as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    # Create dataset_info.json
    dataset_info = {
        "gemma3_vlm_train": {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        }
    }
    
    with open(output_path / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"✓ Saved {len(converted_data)} samples to {output_path}")
    print(f"  - train.json")
    print(f"  - dataset_info.json")
    print(f"  - images/")
    return output_path


def download_pokemon_for_llamafactory(output_dir: str, num_samples: int = 500):
    """Download Pokemon captions in LLaMA-Factory format"""
    from datasets import load_dataset
    
    print("Downloading Pokemon-BLIP-Captions for LLaMA-Factory...")
    
    dataset = load_dataset("lambdalabs/pokemon-blip-captions", split="train")
    
    if num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
    
    output_path = Path(output_dir)
    images_path = output_path / "images"
    output_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(exist_ok=True)
    
    converted_data = []
    
    for idx, item in enumerate(tqdm(dataset, desc="Processing")):
        try:
            image = item["image"]
            caption = item["text"]
            
            image_filename = f"pokemon_{idx:06d}.png"
            image_path = images_path / image_filename
            image.save(image_path)
            
            converted_data.append({
                "messages": [
                    {"role": "user", "content": "<image>What Pokemon is this? Describe it."},
                    {"role": "assistant", "content": caption}
                ],
                "images": [f"images/{image_filename}"]
            })
        except Exception as e:
            print(f"Warning: Failed item {idx}: {e}")
            continue
    
    with open(output_path / "train.json", "w", encoding="utf-8") as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    dataset_info = {
        "gemma3_vlm_train": {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role", "content_tag": "content",
                "user_tag": "user", "assistant_tag": "assistant"
            }
        }
    }
    
    with open(output_path / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"✓ Saved {len(converted_data)} samples to {output_path}")
    return output_path


def download_food_for_llamafactory(output_dir: str, num_samples: int = 500):
    """Download Food-101 in LLaMA-Factory format"""
    from datasets import load_dataset
    
    print("Downloading Food-101 for LLaMA-Factory...")
    
    dataset = load_dataset("ethz/food101", split="train")
    
    if num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
    
    output_path = Path(output_dir)
    images_path = output_path / "images"
    output_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(exist_ok=True)
    
    label_names = dataset.features["label"].names
    converted_data = []
    
    for idx, item in enumerate(tqdm(dataset, desc="Processing")):
        try:
            image = item["image"]
            label = item["label"]
            label_name = label_names[label].replace("_", " ")
            
            image_filename = f"food_{idx:06d}.jpg"
            image_path = images_path / image_filename
            image.save(image_path)
            
            converted_data.append({
                "messages": [
                    {"role": "user", "content": "<image>What food is shown in this image?"},
                    {"role": "assistant", "content": f"This image shows {label_name}."}
                ],
                "images": [f"images/{image_filename}"]
            })
        except Exception as e:
            print(f"Warning: Failed item {idx}: {e}")
            continue
    
    with open(output_path / "train.json", "w", encoding="utf-8") as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    dataset_info = {
        "gemma3_vlm_train": {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role", "content_tag": "content",
                "user_tag": "user", "assistant_tag": "assistant"
            }
        }
    }
    
    with open(output_path / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"✓ Saved {len(converted_data)} samples to {output_path}")
    return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download data for LLaMA-Factory VLM training")
    parser.add_argument("--dataset", choices=["naruto", "pokemon", "food", "all"], default="naruto")
    parser.add_argument("--output_dir", default="data", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=500)
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("LLaMA-Factory Dataset Downloader")
    print("="*60)
    
    if args.dataset in ["naruto", "all"]:
        download_naruto_for_llamafactory(args.output_dir, args.num_samples)
    
    if args.dataset in ["pokemon", "all"]:
        download_pokemon_for_llamafactory(args.output_dir, args.num_samples)
    
    if args.dataset in ["food", "all"]:
        download_food_for_llamafactory(args.output_dir, args.num_samples)
    
    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)
    print("\nTo train with LLaMA-Factory:")
    print(f"  llamafactory-cli train configs/gemma3_projector_sft.yaml")
    print("="*60)


if __name__ == "__main__":
    main()

