#!/bin/bash

declare -a backbone=('resnet-50' 'swin-base' 'mobilenet_v2' 'convnext-base-224') 

d='cifar100'
sd=42

b='resnet-50'
if [ ! -f ../../checkpoint/"${d}"/"${b}"/all_results.json ]; then
    echo ../../checkpoint/"${d}"/"${b}" "Not Found"
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/data_w/jiali/M3U:$PYTHONPATH python train.py configs/train/"${b}".json &
fi

b='vit-base'
if [ ! -f ../../checkpoint/"${d}"/"${b}"/all_results.json ]; then
    echo ../../checkpoint/"${d}"/"${b}" "Not Found"
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/data_w/jiali/M3U:$PYTHONPATH python train.py configs/train/"${b}".json &
fi

b='vit-large'
if [ ! -f ../../checkpoint/"${d}"/"${b}"/all_results.json ]; then
    echo ../../checkpoint/"${d}"/"${b}" "Not Found"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data_w/jiali/M3U:$PYTHONPATH python train.py configs/train/"${b}".json &
fi

b='swin-tiny'
if [ ! -f ../../checkpoint/"${d}"/"${b}"/all_results.json ]; then
    echo ../../checkpoint/"${d}"/"${b}" "Not Found"
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/data_w/jiali/M3U:$PYTHONPATH python train.py configs/train/"${b}".json &
fi

b='swin-base'
if [ ! -f ../../checkpoint/"${d}"/"${b}"/all_results.json ]; then
    echo ../../checkpoint/"${d}"/"${b}" "Not Found"
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/data_w/jiali/M3U:$PYTHONPATH python train.py configs/train/"${b}".json &
fi

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait

