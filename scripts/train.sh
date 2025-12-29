#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:${pwd}

# llamafactory-cli train configs/gemma3_projector_sft.yaml

llamafactory-cli train configs/gemma3_trans_projector_head_sft.yaml