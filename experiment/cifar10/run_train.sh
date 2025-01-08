#!/bin/bash

declare -a backbone=('resnet-34' 'resnet-50' 'swin-base' 'swin-tiny' 'vit-base') 

d='cifar10'
sd=42

for b in "${backbone[@]}"
do
    if [ ! -f ../../checkpoint/"${d}"/"${b}"/all_results.json ]; then
        echo ../../checkpoint/"${d}"/"${b}" "Not Found"
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data_w/jiali/M3U:$PYTHONPATH python train.py configs/train/"${b}".json &
    fi
    trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
    wait
done

