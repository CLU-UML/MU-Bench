#!/bin/bash

declare -a backbone=('llama2-7b')
declare -a method=('neggrad' 'random_label' 'grad_diff' 'npo')
declare -a dfratio=('1' '5' '10')
declare -a seed=('42' '87' '21' '13' '100')
declare -a so_info=('' '_so')

d='tofu'

set -e

for b in "${backbone[@]}"
do
    for m in "${method[@]}"
    do
        for df in "${dfratio[@]}"
        do
            for so in "${so_info[@]}"
            do
            # for sd in "${seed[@]}"
            # do
                sd=42
                if [ ! -f ../../checkpoint/unlearn/"${d}"/"${b}"/"${m}""${so}"/"${df}"/"${sd}"/all_results.json ]; then
                    echo ../../checkpoint/unlearn/"${d}"/"${b}"/"${m}""${so}"/"${df}"/"${sd}" "Not Found"
                    WANDB_MODE=offline CUDA_VISIBLE_DEVICES=4 PYTHONPATH=../..:$PYTHONPATH python unlearn.py configs/unlearn/"${b}"/"${m}""${so}"/"${df}"/"${sd}".json &
                fi

                sd=87
                if [ ! -f ../../checkpoint/unlearn/"${d}"/"${b}"/"${m}""${so}"/"${df}"/"${sd}"/all_results.json ]; then
                    echo ../../checkpoint/unlearn/"${d}"/"${b}"/"${m}""${so}"/"${df}"/"${sd}" "Not Found"
                    WANDB_MODE=offline CUDA_VISIBLE_DEVICES=5 PYTHONPATH=../..:$PYTHONPATH python unlearn.py configs/unlearn/"${b}"/"${m}""${so}"/"${df}"/"${sd}".json &
                fi

                trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
                wait
            # done
            done
        done
    done
done

# Unlearn CL
for b in "${backbone[@]}"
do
    for m in "${method[@]}"
    do
        for so in "${so_info[@]}"
        do
            # for df in "${dfratio[@]}"
            # do
            # for sd in "${seed[@]}"
            # do
            sd=42
            df=1
            if [ ! -f ../../checkpoint/unlearn_cl/"${d}"/"${b}"/"${m}""${so}"/"${df}"/"${sd}"/all_results.json ]; then
                echo ../../checkpoint/unlearn_cl/"${d}"/"${b}"/"${m}""${so}"/"${df}"/"${sd}" "Not Found"
                WANDB_MODE=offline CUDA_VISIBLE_DEVICES=5 PYTHONPATH=../..:$PYTHONPATH python unlearn.py configs/unlearn_cl/"${b}"/"${m}""${so}"/"${df}"/"${sd}".json &
            fi

            df=5
            if [ ! -f ../../checkpoint/unlearn_cl/"${d}"/"${b}"/"${m}""${so}"/"${df}"/"${sd}"/all_results.json ]; then
                echo ../../checkpoint/unlearn_cl/"${d}"/"${b}"/"${m}""${so}"/"${df}"/"${sd}" "Not Found"
                WANDB_MODE=offline CUDA_VISIBLE_DEVICES=6 PYTHONPATH=../..:$PYTHONPATH python unlearn.py configs/unlearn_cl/"${b}"/"${m}""${so}"/"${df}"/"${sd}".json &
            fi

            df=10
            if [ ! -f ../../checkpoint/unlearn_cl/"${d}"/"${b}"/"${m}""${so}"/"${df}"/"${sd}"/all_results.json ]; then
                echo ../../checkpoint/unlearn_cl/"${d}"/"${b}"/"${m}""${so}"/"${df}"/"${sd}" "Not Found"
                WANDB_MODE=offline CUDA_VISIBLE_DEVICES=7 PYTHONPATH=../..:$PYTHONPATH python unlearn.py configs/unlearn_cl/"${b}"/"${m}""${so}"/"${df}"/"${sd}".json &
            fi

            # sd=87
            # if [ ! -f ../../checkpoint/unlearn_cl/"${d}"/"${b}"/"${m}""${so}"/"${df}"/"${sd}"/all_results.json ]; then
            #     echo ../../checkpoint/unlearn_cl/"${d}"/"${b}"/"${m}""${so}"/"${df}"/"${sd}" "Not Found"
            #     WANDB_MODE=offline CUDA_VISIBLE_DEVICES=7 PYTHONPATH=../..:$PYTHONPATH python unlearn.py configs/unlearn_cl/"${b}"/"${m}""${so}"/"${df}"/"${sd}".json &
            # fi

            trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
            wait
            # done
        done
    done
done
