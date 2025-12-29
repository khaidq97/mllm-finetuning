# MLLM Fine-tuning Guide

A comprehensive guide for fine-tuning Multimodal Large Language Models (MLLM) using LLaMA-Factory, specifically for Gemma3 Vision Language Model.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Fine-tuning](#fine-tuning)
- [Testing](#testing)
- [Project Structure](#project-structure)

## Overview

This project provides a framework for fine-tuning vision-language models (VLM) with support for:
- **Projector-only training**: Train only the multi-modal projector while freezing the vision encoder and language model
- **Partial language model training**: Optionally train the last N layers of the language model
- **Flexible training strategies**: Support for LoRA, full fine-tuning, or freeze-based methods

## Quick Start

Get started with fine-tuning in 4 simple steps:

```bash
# 1. Install LLaMA-Factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install ".[metrics]"

# 2. Download sample data (Naruto anime dataset)
cd /path/to/mllm-finetuning
python tools/download_data.py --dataset naruto --num_train_samples 500 --num_val_samples 50

# 3. Start training
llamafactory-cli train configs/gemma3_trans_projector_head_sft.yaml

# 4. Test your model
python tools/test.py
```

**Note:** Make sure you have a pretrained Gemma3 model in `pretrains/gemma3-4b-it/` before training.

## Installation

### Step 1: Install LLaMA-Factory

First, clone and install LLaMA-Factory:

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install ".[metrics]"
```

This will install LLaMA-Factory with all necessary dependencies including metrics support.

### Step 2: Install Additional Dependencies

```bash
# Install project requirements
pip install -r requirements.txt

# Install accelerate for distributed training
pip install accelerate
```

### Step 3: Download Sample Data

Use the provided download script to get sample datasets:

```bash
# Download Naruto dataset (default, recommended for demo)
python tools/download_data.py --dataset naruto --num_train_samples 500 --num_val_samples 50

# Or download Pokemon dataset
python tools/download_data.py --dataset pokemon --num_train_samples 500 --num_val_samples 50

# Or download Food-101 dataset
python tools/download_data.py --dataset food --num_train_samples 500 --num_val_samples 50

# Or download all datasets
python tools/download_data.py --dataset all --num_train_samples 500 --num_val_samples 50
```

**Download Script Options:**
- `--dataset`: Choose dataset type
  - `naruto`: Anime-style images with captions (recommended for demo)
  - `pokemon`: Pokemon images with descriptions
  - `food`: Food-101 dish images with labels
  - `all`: Download all available datasets
- `--output_dir`: Output directory (default: `data`)
- `--num_train_samples`: Number of training samples (default: 500)
- `--num_val_samples`: Number of validation samples (default: 50)

**What the download script does:**
1. Downloads images from HuggingFace datasets
2. Creates train/val splits automatically
3. Generates `train.json` and `val.json` in LLaMA-Factory format
4. Creates `dataset_info.json` with proper configuration
5. Saves all images to `data/images/` directory

**Example output structure after download:**
```
data/
├── dataset_info.json
├── train.json (500 samples)
├── val.json (50 samples)
└── images/
    ├── naruto_train_000000.png
    ├── naruto_train_000001.png
    ├── ...
    ├── naruto_val_000000.png
    └── ...
```

## Data Preparation

### Option 1: Download Sample Datasets (Recommended for Getting Started)

The easiest way to get started is to use the provided download script:

```bash
python tools/download_data.py --dataset naruto --num_train_samples 500 --num_val_samples 50
```

**Available Datasets:**

1. **Naruto Dataset** (Recommended for demo)
   - Anime-style images with detailed captions
   - Source: `lambdalabs/naruto-blip-captions`
   - Varied prompts for better generalization

2. **Pokemon Dataset**
   - Pokemon character images with descriptions
   - Source: `lambdalabs/pokemon-blip-captions`
   - Good for character recognition tasks

3. **Food-101 Dataset**
   - Food dish images with labels
   - Source: `ethz/food101`
   - 101 different food categories

**Download Script Usage:**

```bash
# Basic usage
python tools/download_data.py --dataset naruto

# Custom sample sizes
python tools/download_data.py --dataset naruto --num_train_samples 1000 --num_val_samples 100

# Download to specific directory
python tools/download_data.py --dataset pokemon --output_dir my_data

# Download all datasets
python tools/download_data.py --dataset all
```

The script automatically:
- Downloads images from HuggingFace
- Creates train/validation splits
- Generates properly formatted JSON files
- Sets up `dataset_info.json` configuration
- Organizes images in the `images/` folder

### Option 2: Prepare Your Own Dataset

#### Dataset Structure

The training data follows the ShareGPT format with image support. Data should be organized as follows:

```
data/
├── dataset_info.json    # Dataset configuration
├── train.json           # Training data
├── val.json             # Validation data
└── images/              # Image files
    ├── image1.png
    ├── image2.jpg
    └── ...
```

### Dataset Configuration (`dataset_info.json`)

Define your datasets in `dataset_info.json`:

```json
{
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
```

### Sample Data Format

Each entry in your `train.json` or `val.json` should follow this structure:

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "<image>Describe this anime image in detail."
      },
      {
        "role": "assistant",
        "content": "a man with glasses and a shirt on"
      }
    ],
    "images": ["images/sample_image.png"]
  }
]
```

**Key Points:**
- The `<image>` token in the user message indicates where the image should be inserted
- The `images` field contains paths to image files relative to the `data/` directory
- Multiple images can be included in a single conversation
- Messages follow a conversational format with alternating `user` and `assistant` roles

### Creating Your Own Custom Dataset

If you want to fine-tune on your own data instead of the sample datasets:

1. **Prepare your images**: Place all images in the `data/images/` directory
   ```bash
   mkdir -p data/images
   cp your_images/* data/images/
   ```

2. **Create annotations**: Format your data according to the sample structure above
   - Create `train.json` with training samples
   - Create `val.json` with validation samples
   - Follow the ShareGPT format with `messages` and `images` fields

3. **Update dataset_info.json**: Register your dataset with appropriate configuration
   ```json
   {
     "your_dataset_name": {
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
   ```

4. **Verify paths**: Ensure image paths in JSON files are relative to the `data/` directory

5. **Update config**: Modify your training config to use your dataset name
   ```yaml
   dataset: your_dataset_name
   eval_dataset: your_eval_dataset_name
   ```

## Fine-tuning

### Configuration

The training configuration is managed through YAML files in the `configs/` directory. Key configuration file: `configs/gemma3_trans_projector_head_sft.yaml`

### Key Configuration Parameters

#### Model Settings
- `model_name_or_path`: Path to pretrained model (e.g., `pretrains/gemma3-4b-it`)
- `trust_remote_code`: Enable for custom model implementations

#### Training Method
- `stage`: Training stage (`sft` for supervised fine-tuning)
- `finetuning_type`: `freeze`, `lora`, or `full`

#### Freeze Settings (Projector-Only Training)
```yaml
freeze_vision_tower: true              # Freeze vision encoder (SigLIP)
freeze_language_model: false           # Allow partial training of LM
freeze_multi_modal_projector: false    # Train projector layer
freeze_trainable_layers: 2             # Train last 2 layers of LM
freeze_extra_modules: multi_modal_projector,post_layernorm
```

#### Dataset Settings
```yaml
dataset_dir: data
dataset: gemma3_vlm_train              # Training dataset name
eval_dataset: gemma3_vlm_val           # Validation dataset name
template: gemma3                       # Prompt template
cutoff_len: 2048                       # Max sequence length
```

#### Training Hyperparameters
```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 2
learning_rate: 2.0e-5
num_train_epochs: 500
lr_scheduler_type: cosine
warmup_ratio: 0.05
```

#### Output Settings
```yaml
output_dir: outputs/gemma3_trans_projector_head_sft
save_steps: 5000                       # Checkpoint frequency
eval_steps: 1000                       # Evaluation frequency
logging_steps: 300
```

### Starting Training

#### Using LLaMA-Factory CLI

```bash
llamafactory-cli train configs/gemma3_trans_projector_head_sft.yaml
```

#### Using Training Script

```bash
bash scripts/train.sh
```

#### Using Web UI

```bash
bash scripts/webui.sh
```

### Monitoring Training

Training logs are saved to:
- **TensorBoard logs**: `outputs/gemma3_trans_projector_head_sft/logs/`
- **Training log**: `outputs/gemma3_trans_projector_head_sft/trainer_log.jsonl`
- **Checkpoints**: `outputs/gemma3_trans_projector_head_sft/checkpoint-{step}/`

View training progress with TensorBoard:
```bash
tensorboard --logdir outputs/gemma3_trans_projector_head_sft/logs
```

## Testing

### Test Script Overview

The `tools/test.py` script allows you to test your fine-tuned model with a sample image and prompt.

### Configuration

Edit the following variables in `tools/test.py`:

```python
# Model checkpoint path
CHECKPOINT_PATH = "/path/to/checkpoint"  # e.g., "outputs/gemma3_trans_projector_head_sft/checkpoint-10000"

# Test image path
IMAGE_PATH = "/path/to/test/image.png"  # e.g., "data/images/sample.png"

# Test prompt
PROMPT = "Describe this image in detail."
```

### Running the Test

```bash
cd /home/sagemaker-user/workspace/mllm-finetuning
python tools/test.py
```

### Test Options

You can test different checkpoints by modifying `CHECKPOINT_PATH`:

1. **Base pretrained model**:
   ```python
   CHECKPOINT_PATH = "pretrains/gemma3-4b-it"
   ```

2. **Fine-tuned checkpoint**:
   ```python
   CHECKPOINT_PATH = "outputs/gemma3_trans_projector_head_sft/checkpoint-10000"
   ```

3. **Latest checkpoint**:
   ```python
   CHECKPOINT_PATH = "outputs/gemma3_trans_projector_head_sft/checkpoint-15000"
   ```

### Expected Output

The script will:
1. Load the model and processor from the checkpoint
2. Process the input image and prompt
3. Generate a response
4. Print the generated text and inference time

Example output:
```
The image shows a vibrant garden scene with pink cosmos flowers and a bumblebee collecting nectar...
Time taken: 2.34 seconds
```

### Customizing Test Prompts

You can modify the conversation structure in `tools/test.py`:

```python
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Your custom prompt here"}
        ]
    }
]
```

### Generation Parameters

Adjust generation settings in the test script:

```python
generation = model.generate(
    **inputs,
    max_new_tokens=100,      # Maximum tokens to generate
    do_sample=False,         # Use greedy decoding (deterministic)
    temperature=0.7,         # Sampling temperature (if do_sample=True)
    top_p=0.9,              # Nucleus sampling parameter
)
```

## Project Structure

```
mllm-finetuning/
├── configs/                           # Training configurations
│   ├── gemma3_projector_sft.yaml
│   └── gemma3_trans_projector_head_sft.yaml
├── data/                              # Dataset directory
│   ├── dataset_info.json
│   ├── train.json
│   ├── val.json
│   └── images/
├── outputs/                           # Training outputs
│   └── gemma3_trans_projector_head_sft/
│       ├── checkpoint-5000/
│       ├── checkpoint-10000/
│       └── trainer_log.jsonl
├── pretrains/                         # Pretrained models
│   ├── gemma3-4b-it/
│   └── gemma3-4b-it-projector/
├── scripts/                           # Helper scripts
│   ├── train.sh
│   └── webui.sh
├── tools/                             # Utilities
│   ├── download_data.py
│   └── test.py
├── README.md
└── requirements.txt
```

## Tips and Best Practices

### Training Tips

1. **Start with projector-only training**: Train only the projector first, then optionally fine-tune additional layers
2. **Monitor validation loss**: Use `eval_strategy: steps` and `eval_steps: 1000` to track overfitting
3. **Checkpoint management**: Set `save_total_limit: 5` to avoid disk space issues
4. **Batch size tuning**: Adjust `per_device_train_batch_size` and `gradient_accumulation_steps` based on GPU memory

### Data Tips

1. **Image quality**: Use high-quality images with clear content
2. **Diverse captions**: Include varied descriptions to improve model generalization
3. **Data augmentation**: Consider augmenting images to increase dataset size
4. **Validation set**: Always keep a separate validation set (10-20% of data)

### Debugging

1. **Test with small dataset**: Use `max_samples: 100` during initial testing
2. **Check data loading**: Verify image paths are correct in JSON files
3. **Monitor GPU memory**: Use `nvidia-smi` to track memory usage
4. **Gradient clipping**: If training is unstable, adjust `max_grad_norm`

## Troubleshooting

### Common Issues

**Issue**: Out of memory during training
- **Solution**: Reduce batch size, enable gradient checkpointing, or use gradient accumulation

**Issue**: Model not improving
- **Solution**: Check learning rate, verify data quality, increase training epochs

**Issue**: Images not loading
- **Solution**: Verify image paths are relative to `data/` directory, check file permissions

**Issue**: Checkpoint loading fails
- **Solution**: Ensure checkpoint directory contains all required files (config.json, model files, tokenizer files)

## License

Please refer to the original model licenses for usage restrictions.

## Acknowledgments

- Built with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- Based on Google's Gemma3 model architecture

