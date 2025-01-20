#!/bin/bash

# 'mobilenet_v2' 'convnext-base-224' 'swin-base')
set -e
d='cifar100'
declare -a backbone=('resnet-50' 'swin-base')
declare -a method=('neggrad' 'random_label' 'bad_teaching' 'salun')
declare -a method=('random_label' 'bad_teaching' 'salun')
declare -a dfratio=('2.0' '4.0' '6.0' '8.0' '10.0')
declare -a seed=('42' '87' '21' '13' '100')
declare -a seed=('42' '87')
declare -a config=('_so')

for b in "${backbone[@]}"
do
    for m in "${method[@]}"
    do
        for sd in "${seed[@]}"
        do
            for cf in "${config[@]}"
            do
                # for df in "${dfratio[@]}"
                # do
                df=2.0
                if [ ! -f ../../checkpoint/unlearn"${cf}"/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}"/a.json ]; then
                    echo ../../checkpoint/unlearn"${cf}"/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}" "Not Found"
                    WANDB_MODE=offline CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../..:$PYTHONPATH python unlearn.py configs/unlearn"${cf}"/"${b}"/"${m}"/"${df}"/"${sd}".json &
                fi

                df=4.0
                if [ ! -f ../../checkpoint/unlearn"${cf}"/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}"/a.json ]; then
                    echo ../../checkpoint/unlearn"${cf}"/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}" "Not Found"
                    WANDB_MODE=offline CUDA_VISIBLE_DEVICES=4 PYTHONPATH=../..:$PYTHONPATH python unlearn.py configs/unlearn"${cf}"/"${b}"/"${m}"/"${df}"/"${sd}".json &
                fi

                df=6.0
                if [ ! -f ../../checkpoint/unlearn"${cf}"/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}"/a.json ]; then
                    echo ../../checkpoint/unlearn"${cf}"/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}" "Not Found"
                    WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2 PYTHONPATH=../..:$PYTHONPATH python unlearn.py configs/unlearn"${cf}"/"${b}"/"${m}"/"${df}"/"${sd}".json &
                fi

                df=8.0
                if [ ! -f ../../checkpoint/unlearn"${cf}"/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}"/a.json ]; then
                    echo ../../checkpoint/unlearn"${cf}"/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}" "Not Found"
                    WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../..:$PYTHONPATH python unlearn.py configs/unlearn"${cf}"/"${b}"/"${m}"/"${df}"/"${sd}".json &
                fi

                df=10.0
                if [ ! -f ../../checkpoint/unlearn"${cf}"/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}"/a.json ]; then
                    echo ../../checkpoint/unlearn"${cf}"/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}" "Not Found"
                    WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../..:$PYTHONPATH python unlearn.py configs/unlearn"${cf}"/"${b}"/"${m}"/"${df}"/"${sd}".json &
                fi

                trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
                wait
                # done
            done
        done
    done
done
