#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate llm-uncertainty
CUDA_VISIBLE_DEVICES=1 python -m llm_linguistic_confidence_study