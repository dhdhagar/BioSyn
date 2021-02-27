#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate biosyn

module load cudnn/7.6-cuda_9.0 cuda90

MODEL=biosyn-ncbi-disease
MODEL_DIR=./tmp/${MODEL}
DATA_DIR=./datasets/ncbi-disease

python inference.py \
    --model_dir ${MODEL_DIR} \
    --dictionary_path ${DATA_DIR}/test_dictionary.txt \
    --use_cuda \
    --mention "ataxia telangiectasia" \
    --show_predictions
