#!/bin/bash

declare -a backbone=('bert-base' 'bert-large' 'roberta-base' 'roberta-large' 'electra-base') 

d='imdb'
sd=42

for b in "${backbone[@]}"
do
    if [ ! -f ../../checkpoint/"${d}"/"${b}"/a.json ]; then
        echo ../../checkpoint/"${d}"/"${b}" "Not Found"
        CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/data_w/jiali/M3U:$PYTHONPATH python train.py configs/train/"${b}".json
    fi
    trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
    wait
done
