#!/bin/bash

set -e
declare -a backbone=('vilt')
declare -a method=('neggrad' 'random_label' 'bad_teaching' 'salun')
declare -a dfratio=('2.0' '4.0' '6.0' '8.0' '10.0')

d='nlvr2'


for b in "${backbone[@]}"
do
    for m in "${method[@]}"
    do
        for df in "${dfratio[@]}"
        do
            # for sd in "${seed[@]}"
            # do
            sd=42
            if [ ! -f checkpoint/unlearn/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}"/a.json ]; then
                echo checkpoint/unlearn/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}" "Not Found"
                WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/data_w/jiali/M3U:$PYTHONPATH python unlearn.py configs/unlearn/"${b}"/"${m}"/"${df}"/"${sd}".json &
            fi

            sd=87
            if [ ! -f checkpoint/unlearn/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}"/a.json ]; then
                echo checkpoint/unlearn/"${d}"/"${b}"/"${m}"/"${df}"/"${sd}" "Not Found"
                WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/data_w/jiali/M3U:$PYTHONPATH python unlearn.py configs/unlearn/"${b}"/"${m}"/"${df}"/"${sd}".json &
            fi

            trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
            wait
            # done
        done
    done
done
