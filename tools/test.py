# pip install accelerate
import time
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch

# CHECKPOINT_PATH = "/home/sagemaker-user/workspace/mllm-finetuning/outputs/gemma3_projector_sft/checkpoint-800"
# CHECKPOINT_PATH = "/home/sagemaker-user/workspace/mllm-finetuning/pretrains/gemma3-4b-it"
CHECKPOINT_PATH = "/home/sagemaker-user/workspace/mllm-finetuning/outputs/gemma3_trans_projector_head_sft/checkpoint-10000"
IMAGE_PATH = "/home/sagemaker-user/workspace/mllm-finetuning/data/images/naruto_val_000022.png"
PROMPT = "Describe this image in detail."

image = Image.open(IMAGE_PATH).convert("RGB")

model = Gemma3ForConditionalGeneration.from_pretrained(
    CHECKPOINT_PATH,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(
    CHECKPOINT_PATH,
    trust_remote_code=True
)   

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]
    }
]

start_time = time.time()
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
# **Overall Impression:** The image is a close-up shot of a vibrant garden scene, 
# focusing on a cluster of pink cosmos flowers and a busy bumblebee. 
# It has a slightly soft, natural feel, likely captured in daylight.
