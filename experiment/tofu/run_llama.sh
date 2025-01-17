#!/bin/bash

declare -a backbone=('llama2-7b')
declare -a method=('neggrad' 'random_label' 'grad_diff' 'npo')
declare -a dfratio=('1' '5' '10')
declare -a seed=('42' '87' '21' '13' '100')

d='tofu'

set -e

for b in "${backbone[@]}"
do
    for m in "${method[@]}"
    do
        for df in "${dfratio[@]}"
        do
            # for sd in "${seed[@]}"
            # do
            sd=42
            if [ ! -f ../../checkpoint/unlearn/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}"/a.json ]; then
                echo ../../checkpoint/unlearn/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}" "Not Found"
                WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=/data_w/jiali/M3U:$PYTHONPATH python unlearn.py configs/unlearn/"${b}"/"${m}"/"${df}"/"${sd}".json 
            fi

            sd=87
            if [ ! -f ../../checkpoint/unlearn/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}"/a.json ]; then
                echo ../../checkpoint/unlearn/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}" "Not Found"
                WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=/data_w/jiali/M3U:$PYTHONPATH python unlearn.py configs/unlearn/"${b}"/"${m}"/"${df}"/"${sd}".json
            fi

            trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
            wait
            # done
        done
    done
done