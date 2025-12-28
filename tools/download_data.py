#!/usr/bin/env python3
"""
Download sample datasets in LLaMA-Factory format for Gemma 3 VLM training
Includes both training and validation splits
"""
import os
import json
from pathlib import Path
from tqdm import tqdm


def download_naruto_for_llamafactory(
    output_dir: str, 
    num_train_samples: int = 500,
    num_val_samples: int = 50,
    val_ratio: float = None
):
    """
    Download Naruto captions in LLaMA-Factory format with train/val split.
    
    Args:
        output_dir: Output directory
        num_train_samples: Number of training samples (default: 500)
        num_val_samples: Number of validation samples (default: 50)
        val_ratio: If set, use this ratio for validation split instead of fixed numbers
    """
    from datasets import load_dataset
    
    print("Downloading Naruto-BLIP-Captions for LLaMA-Factory...")
    print(f"  Train samples: {num_train_samples}")
    print(f"  Val samples: {num_val_samples}")
    
    dataset = load_dataset("lambdalabs/naruto-blip-captions", split="train")
    
    # Calculate split sizes
    total_available = len(dataset)
    
    if val_ratio is not None:
        total_samples = min(num_train_samples + num_val_samples, total_available)
        num_val_samples = int(total_samples * val_ratio)
        num_train_samples = total_samples - num_val_samples
    
    total_needed = num_train_samples + num_val_samples
    if total_needed > total_available:
        print(f"Warning: Requested {total_needed} samples but only {total_available} available")
        ratio = num_train_samples / total_needed
        num_train_samples = int(total_available * ratio)
        num_val_samples = total_available - num_train_samples
    
    # Shuffle and split
    dataset = dataset.shuffle(seed=42)
    train_dataset = dataset.select(range(num_train_samples))
    val_dataset = dataset.select(range(num_train_samples, num_train_samples + num_val_samples))
    
    # Create output directories
    output_path = Path(output_dir)
    images_path = output_path / "images"
    output_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(exist_ok=True)
    
    # Process training data
    print("\nProcessing training data...")
    train_data = _process_naruto_split(train_dataset, images_path, prefix="train")
    
    # Process validation data
    print("\nProcessing validation data...")
    val_data = _process_naruto_split(val_dataset, images_path, prefix="val")
    
    # Save JSON files
    with open(output_path / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(output_path / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    # Create dataset_info.json with both train and val datasets
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
        },
        "gemma3_vlm_val": {
            "file_name": "val.json",
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
    
    print(f"\n✓ Saved datasets to {output_path}")
    print(f"  - train.json: {len(train_data)} samples")
    print(f"  - val.json: {len(val_data)} samples")
    print(f"  - dataset_info.json")
    print(f"  - images/")
    
    return output_path


def _process_naruto_split(dataset, images_path, prefix="train"):
    """Process a split of the Naruto dataset"""
    converted_data = []
    
    for idx, item in enumerate(tqdm(dataset, desc=f"Processing {prefix}")):
        try:
            image = item["image"]
            caption = item["text"]
            
            image_filename = f"naruto_{prefix}_{idx:06d}.png"
            image_path = images_path / image_filename
            image.save(image_path)
            
            # LLaMA-Factory format with varied prompts for better generalization
            prompts = [
                "Describe this anime image in detail.",
                "What do you see in this Naruto-style image?",
                "Describe the character and scene in this image.",
                "What is shown in this anime illustration?",
                "Provide a detailed description of this image.",
            ]
            prompt = prompts[idx % len(prompts)]
            
            converted_data.append({
                "messages": [
                    {"role": "user", "content": f"<image>{prompt}"},
                    {"role": "assistant", "content": caption}
                ],
                "images": [f"images/{image_filename}"]
            })
        except Exception as e:
            print(f"Warning: Failed {prefix} item {idx}: {e}")
            continue
    
    return converted_data


def download_pokemon_for_llamafactory(
    output_dir: str, 
    num_train_samples: int = 500,
    num_val_samples: int = 50
):
    """Download Pokemon captions in LLaMA-Factory format with train/val split"""
    from datasets import load_dataset
    
    print("Downloading Pokemon-BLIP-Captions for LLaMA-Factory...")
    print(f"  Train samples: {num_train_samples}")
    print(f"  Val samples: {num_val_samples}")
    
    dataset = load_dataset("lambdalabs/pokemon-blip-captions", split="train")
    
    total_available = len(dataset)
    total_needed = num_train_samples + num_val_samples
    if total_needed > total_available:
        ratio = num_train_samples / total_needed
        num_train_samples = int(total_available * ratio)
        num_val_samples = total_available - num_train_samples
    
    dataset = dataset.shuffle(seed=42)
    train_dataset = dataset.select(range(num_train_samples))
    val_dataset = dataset.select(range(num_train_samples, num_train_samples + num_val_samples))
    
    output_path = Path(output_dir)
    images_path = output_path / "images"
    output_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(exist_ok=True)
    
    # Process training data
    print("\nProcessing training data...")
    train_data = _process_pokemon_split(train_dataset, images_path, prefix="train")
    
    # Process validation data
    print("\nProcessing validation data...")
    val_data = _process_pokemon_split(val_dataset, images_path, prefix="val")
    
    with open(output_path / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(output_path / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    dataset_info = {
        "gemma3_vlm_train": {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role", "content_tag": "content",
                "user_tag": "user", "assistant_tag": "assistant"
            }
        },
        "gemma3_vlm_val": {
            "file_name": "val.json",
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
    
    print(f"\n✓ Saved datasets to {output_path}")
    print(f"  - train.json: {len(train_data)} samples")
    print(f"  - val.json: {len(val_data)} samples")
    return output_path


def _process_pokemon_split(dataset, images_path, prefix="train"):
    """Process a split of the Pokemon dataset"""
    converted_data = []
    
    for idx, item in enumerate(tqdm(dataset, desc=f"Processing {prefix}")):
        try:
            image = item["image"]
            caption = item["text"]
            
            image_filename = f"pokemon_{prefix}_{idx:06d}.png"
            image_path = images_path / image_filename
            image.save(image_path)
            
            prompts = [
                "What Pokemon is this? Describe it.",
                "Describe this Pokemon image in detail.",
                "What creature is shown in this image?",
                "Identify and describe this Pokemon.",
                "What do you see in this Pokemon illustration?",
            ]
            prompt = prompts[idx % len(prompts)]
            
            converted_data.append({
                "messages": [
                    {"role": "user", "content": f"<image>{prompt}"},
                    {"role": "assistant", "content": caption}
                ],
                "images": [f"images/{image_filename}"]
            })
        except Exception as e:
            print(f"Warning: Failed {prefix} item {idx}: {e}")
            continue
    
    return converted_data


def download_food_for_llamafactory(
    output_dir: str, 
    num_train_samples: int = 500,
    num_val_samples: int = 50
):
    """Download Food-101 in LLaMA-Factory format with train/val split"""
    from datasets import load_dataset
    
    print("Downloading Food-101 for LLaMA-Factory...")
    print(f"  Train samples: {num_train_samples}")
    print(f"  Val samples: {num_val_samples}")
    
    dataset = load_dataset("ethz/food101", split="train")
    
    total_available = len(dataset)
    total_needed = num_train_samples + num_val_samples
    if total_needed > total_available:
        ratio = num_train_samples / total_needed
        num_train_samples = int(total_available * ratio)
        num_val_samples = total_available - num_train_samples
    
    dataset = dataset.shuffle(seed=42)
    train_dataset = dataset.select(range(num_train_samples))
    val_dataset = dataset.select(range(num_train_samples, num_train_samples + num_val_samples))
    
    output_path = Path(output_dir)
    images_path = output_path / "images"
    output_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(exist_ok=True)
    
    label_names = dataset.features["label"].names
    
    print("\nProcessing training data...")
    train_data = _process_food_split(train_dataset, images_path, label_names, prefix="train")
    
    print("\nProcessing validation data...")
    val_data = _process_food_split(val_dataset, images_path, label_names, prefix="val")
    
    with open(output_path / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(output_path / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    dataset_info = {
        "gemma3_vlm_train": {
            "file_name": "train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role", "content_tag": "content",
                "user_tag": "user", "assistant_tag": "assistant"
            }
        },
        "gemma3_vlm_val": {
            "file_name": "val.json",
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
    
    print(f"\n✓ Saved datasets to {output_path}")
    print(f"  - train.json: {len(train_data)} samples")
    print(f"  - val.json: {len(val_data)} samples")
    return output_path


def _process_food_split(dataset, images_path, label_names, prefix="train"):
    """Process a split of the Food dataset"""
    converted_data = []
    
    for idx, item in enumerate(tqdm(dataset, desc=f"Processing {prefix}")):
        try:
            image = item["image"]
            label = item["label"]
            label_name = label_names[label].replace("_", " ")
            
            image_filename = f"food_{prefix}_{idx:06d}.jpg"
            image_path = images_path / image_filename
            image.save(image_path)
            
            prompts = [
                "What food is shown in this image?",
                "Identify the dish in this photo.",
                "What is this food?",
                "Describe the food item in this image.",
                "What dish can you see here?",
            ]
            prompt = prompts[idx % len(prompts)]
            
            converted_data.append({
                "messages": [
                    {"role": "user", "content": f"<image>{prompt}"},
                    {"role": "assistant", "content": f"This image shows {label_name}."}
                ],
                "images": [f"images/{image_filename}"]
            })
        except Exception as e:
            print(f"Warning: Failed {prefix} item {idx}: {e}")
            continue
    
    return converted_data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download data for LLaMA-Factory VLM training (with train/val split)"
    )
    parser.add_argument(
        "--dataset", 
        choices=["naruto", "pokemon", "food", "all"], 
        default="naruto",
        help="Dataset to download (default: naruto - recommended for finetuning demo)"
    )
    parser.add_argument("--output_dir", default="data", help="Output directory")
    parser.add_argument(
        "--num_train_samples", 
        type=int, 
        default=500,
        help="Number of training samples (default: 500)"
    )
    parser.add_argument(
        "--num_val_samples", 
        type=int, 
        default=50,
        help="Number of validation samples (default: 50)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("LLaMA-Factory Dataset Downloader (Train + Validation)")
    print("="*60)
    
    if args.dataset in ["naruto", "all"]:
        download_naruto_for_llamafactory(
            args.output_dir, 
            args.num_train_samples,
            args.num_val_samples
        )
    
    if args.dataset in ["pokemon", "all"]:
        download_pokemon_for_llamafactory(
            args.output_dir, 
            args.num_train_samples,
            args.num_val_samples
        )
    
    if args.dataset in ["food", "all"]:
        download_food_for_llamafactory(
            args.output_dir, 
            args.num_train_samples,
            args.num_val_samples
        )
    
    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)
    print("\nTo train with LLaMA-Factory:")
    print("  llamafactory-cli train configs/gemma3_projector_sft.yaml")
    print("\nDataset info:")
    print("  - Training: gemma3_vlm_train")
    print("  - Validation: gemma3_vlm_val")
    print("="*60)


if __name__ == "__main__":
    main()
