#!/bin/bash

declare -a backbone=('bert-base' 'roberta-base' 'deberta-base' 'distilbert-base' 'electra-base' 'albert-base-v2')
declare -a backbone=('bert-base' 'electra-base')
declare -a method=('neggrad')
declare -a dfratio=('2.0' '4.0' '6.0' '8.0' '10.0')
declare -a seed=('42' '87' '21' '13' '100')

d='imdb'

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
                WANDB_MODE=offline CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/data_w/jiali/M3U:$PYTHONPATH python unlearn.py configs/unlearn/"${b}"/"${m}"/"${df}"/"${sd}".json &
            fi

            sd=87
            if [ ! -f ../../checkpoint/unlearn/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}"/a.json ]; then
                echo ../../checkpoint/unlearn/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}" "Not Found"
                WANDB_MODE=offline CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/data_w/jiali/M3U:$PYTHONPATH python unlearn.py configs/unlearn/"${b}"/"${m}"/"${df}"/"${sd}".json &
            fi

            trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
            wait
            # done
        done
    done
done