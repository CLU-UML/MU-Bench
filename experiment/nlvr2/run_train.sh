#!/bin/bash

declare -a backbone=('vilt') 

d='nlvr2'
sd=42

for b in "${backbone[@]}"
do
    if [ ! -f ../../checkpoint/"${d}"/"${b}"/a.json ]; then
        echo ../../checkpoint/"${d}"/"${b}" "Not Found"
        CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/data_w/jiali/M3U:$PYTHONPATH python train.py configs/train/"${b}".json
    fi
    trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
    wait
done
