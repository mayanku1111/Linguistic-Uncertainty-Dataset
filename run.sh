#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate llm-uncertainty
python -m llm_linguistic_confidence_study