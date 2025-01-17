#!/bin/bash

# 'mobilenet_v2' 'convnext-base-224' 'swin-base')
set -e
declare -a backbone=('resnet-50' 'swin-base')
declare -a method=('neggrad' 'random_label' 'bad_teaching' 'salun')
declare -a dfratio=('2.0' '4.0' '6.0' '8.0' '10.0')
declare -a seed=('42' '87' '21' '13' '100')

d='cifar100'

for b in "${backbone[@]}"
do
    for m in "${method[@]}"
    do
        for df in "${dfratio[@]}"
        do
            # for sd in "${seed[@]}"
            # do
            sd=42
            if [ ! -f checkpoint/unlearn/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}"/all_results.json ]; then
                echo checkpoint/unlearn/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}" "Not Found"
                WANDB_MODE=offline CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../..:$PYTHONPATH python unlearn.py configs/unlearn/"${b}"/"${m}"/"${df}"/"${sd}".json &
            fi

            sd=87
            if [ ! -f checkpoint/unlearn/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}"/all_results.json ]; then
                echo checkpoint/unlearn/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}" "Not Found"
                WANDB_MODE=offline CUDA_VISIBLE_DEVICES=4 PYTHONPATH=../..:$PYTHONPATH python unlearn.py configs/unlearn/"${b}"/"${m}"/"${df}"/"${sd}".json &
            fi

            trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
            wait
            # done
        done
    done
done

for b in "${backbone[@]}"
do
    for m in "${method[@]}"
    do
        for df in "${dfratio[@]}"
        do
            # for sd in "${seed[@]}"
            # do
            sd=42
            if [ ! -f checkpoint/unlearn_cl/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}"/all_results.json ]; then
                echo checkpoint/unlearn_cl/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}" "Not Found"
                WANDB_MODE=offline CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../..:$PYTHONPATH python unlearn.py configs/unlearn_cl/"${b}"/"${m}"/"${df}"/"${sd}".json &
            fi

            sd=87
            if [ ! -f checkpoint/unlearn_cl/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}"/all_results.json ]; then
                echo checkpoint/unlearn_cl/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}" "Not Found"
                WANDB_MODE=offline CUDA_VISIBLE_DEVICES=4 PYTHONPATH=../..:$PYTHONPATH python unlearn.py configs/unlearn_cl/"${b}"/"${m}"/"${df}"/"${sd}".json &
            fi

            trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
            wait
            # done
        done
    done
done
