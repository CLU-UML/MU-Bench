import os
import copy
import json
import argparse


d = 'tofu'
seeds = [42, 87, 21, 100, 13]
backbones = ['phi-1.5', 'llama2-7b']
del_ratio = [1, 5, 10]

general_methods = ['retrain', 'ft', 'neggrad', 'random_label', 'scrub', 'grad_diff', 'npo']
methods = [] + general_methods


def get_full_model_name(m):
    if m == 'phi-1.5':
        m = 'microsoft/phi-1_5'

    elif m == 'llama2-7b':
        m = 'NousResearch/Llama-2-7b-chat-hf'

    return m


template = {
    "do_train": True,
    "do_eval": False,
    "dataset_name": "tofu",
    "num_train_epochs": 5,
    "logging_steps": 10,
    "evaluation_strategy": "no",
    "save_strategy": "epoch",
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "bf16": True,
    "bf16_full_eval": True,
    "optim": "paged_adamw_32bit",
    "learning_rate": 1e-5,
    "warmup_ratio": 0,
    "save_total_limit": 1,
    "metric_for_best_model": "rougeL",
    "overwrite_output_dir": True,
    "dataloader_num_workers": 16,
    "remove_unused_columns": False,
    "model_name_or_path": 'microsoft/phi-1_5',
    'use_so_info': False,
    'use_lora': True,
}


# Unlearn
for dr in del_ratio:
    for m in methods:
        for b in backbones:
            for s in seeds:
                for so in [False, True]:
                    config = copy.deepcopy(template)
                    out_dir = f'{b}/{m + "_so" if so else m}/{dr}'
                    out_name = f'{s}'
                    os.makedirs(f'configs/unlearn/{out_dir}', exist_ok=True)

                    config['unlearn_method'] = m
                    config['del_ratio'] = dr
                    config['model_name_or_path'] = get_full_model_name(b)
                    config['dataset_name'] = d
                    config['warmup_ratio'] = 1 / config['num_train_epochs']
                    config['seed'] = s

                    config['use_cl'] = False
                    config['use_so_info'] = True if so else False
                    config['output_dir'] = f'../../checkpoint/unlearn/{d}/{out_dir}/{out_name}'
                    config['hub_model_id'] = f'{d}-{b}-{m + "_so" if so else m}-{dr}-{s}'

                    with open(f'configs/unlearn/{out_dir}/{out_name}.json', 'w') as f:
                        json.dump(config, f, indent=4)

# Unlearn for CL
for dr in del_ratio:
    for m in methods:
        for b in backbones:
            for s in seeds:
                for so in [False, True]:
                    config = copy.deepcopy(template)
                    out_dir = f'{b}/{m + "_so" if so else m}/{dr}'
                    out_name = f'{s}'
                    os.makedirs(f'configs/unlearn_cl/{out_dir}', exist_ok=True)

                    config['unlearn_method'] = m
                    config['del_ratio'] = dr
                    config['model_name_or_path'] = get_full_model_name(b)
                    config['dataset_name'] = d
                    config['warmup_ratio'] = 1 / config['num_train_epochs']
                    config['seed'] = s

                    config['use_cl'] = True
                    config['use_so_info'] = True if so else False
                    config['output_dir'] = f'../../checkpoint/unlearn_cl/{d}/{out_dir}/{out_name}'
                    config['hub_model_id'] = f'{d}-{b}-{m + "_so" if so else m}-{dr}-{s}'

                    with open(f'configs/unlearn_cl/{out_dir}/{out_name}.json', 'w') as f:
                        json.dump(config, f, indent=4)
