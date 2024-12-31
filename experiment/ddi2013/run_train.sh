#!/bin/bash

declare -a backbone=('biobert' 'pubmedbert-abstract' 'pubmedbert-fulltext') 

d='ddi'
sd=42

# for b in "${backbone[@]}"
# do
# b='biobert'
# if [ ! -f ../../checkpoint/"${d}"/"${b}"/a.json ]; then
#     echo ../../checkpoint/"${d}"/"${b}" "Not Found"
#     CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/data_w/jiali/M3U:$PYTHONPATH python train.py configs/train/"${b}".json &
# fi

b='pubmedbert-fulltext'
if [ ! -f ../../checkpoint/"${d}"/"${b}"/a.json ]; then
    echo ../../checkpoint/"${d}"/"${b}" "Not Found"
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/data_w/jiali/M3U:$PYTHONPATH python train.py configs/train/"${b}".json &
fi

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait
# done
