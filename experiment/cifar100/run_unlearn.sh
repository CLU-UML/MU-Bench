#!/bin/bash

declare -a backbone=('resnet-50' 'swin-base' 'mobilenet_v2' 'convnext-base-224')
declare -a method=('neggrad' 'random_label' 'bad_teaching' 'scrub' 'salul')
declare -a dfratio=('2.0' '4.0' '6.0' '8.0' '10.0')
declare -a seed=('42' '87' '21' '13' '100')

d='cifar100'

for b in "${backbone[@]}"
do
    for m in "${method[@]}"
    do
        for df in "${dfratio[@]}"
        do
            for sd in "${seed[@]}"
            do
                if [ ! -f ../../checkpoint/unlearn/"${d}"/"${b}"/"${m}"/"${df}"/"${d}"_"${sd}"/all_results.json ]; then
                    echo ../../checkpoint/unlearn/"${d}"/"${b}"/"${m}"/"${df}"/"${d}"_"${sd}" "Not Found"
                    WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data_w/jiali/M3U:$PYTHONPATH python unlearn.py configs/unlearn/"${b}"/"${m}"/"${df}"/"${d}"_"${sd}".json
                fi
                trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
                wait
            done
        done
    done
done
