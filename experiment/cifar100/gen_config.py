import os
import copy
import json
import argparse


d = 'cifar100'
seeds = [42, 87, 21, 100, 13]
del_ratio = [2.0, 4.0, 6.0, 8.0, 10.0]
backbones = ['resnet-18', 'resnet-34', 'resnet-50', 'vit-base-patch16-224', 'vit-large-patch16-224', 
             'swin-tiny', 'swin-base', 'mobilenet_v1', 'mobilenet_v2', 'convnext-base-224', 'convnext-base-224-22k']
methods = ['retrain', 'neggrad', 'random_label', 'bad_teaching', 'scrub', 'salul']


def get_full_model_name(m):
    if 'vit-' in m:
        m = 'google/' + m + '-in21k'

    elif 'resnet-' in m:
        m = 'microsoft/' + m

    elif 'swin-' in m:
        m = 'microsoft/' + m + '-patch4-window7-224'

    elif 'mobilenet_' in m:
        m = 'google/' + m + '_1.0_224'

    elif 'convnext' in m:
        m = 'facebook/' + m

    return m

template = {
    "do_train": True,
    "do_eval": True,
    "dataset_name": "mnist",
    "num_train_epochs": 20,
    "logging_steps": 1000,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 256,
    "learning_rate": 2e-5,
    "warmup_steps": 0,
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
    "overwrite_output_dir": True,
    "dataloader_num_workers": 16,
    "ignore_mismatched_sizes": True,
    "remove_unused_columns": False,
    "image_column_name": 'image',
    "label_column_name": 'label',
}


# Train
s = 42
for b in backbones:
    config = copy.deepcopy(template)
    out_dir = f'{b}'
    out_name = f'{d}_{s}'
    os.makedirs(f'configs/train/{out_dir}', exist_ok=True)
    
    config['model_name_or_path'] = get_full_model_name(b)
    config['dataset_name'] = d
    config['seed'] = s
    config['output_dir'] = f'checkpoint/train/{out_dir}/{out_name}'
    config['hub_model_id'] = f'{b}-{d}-{s}'

    with open(f'configs/train/{out_dir}/{out_name}.json', 'w') as f:
        json.dump(config, f, indent=4)

# Unlearn
for b in backbones:
    for s in seeds:
        for dr in del_ratio:
            for m in methods:
                config = copy.deepcopy(template)
                out_dir = f'{b}/{m}/{dr}'
                out_name = f'{d}_{s}'
                os.makedirs(f'configs/unlearn/{out_dir}', exist_ok=True)

                config['unlearn_method'] = m
                config['del_ratio'] = dr
                if m == 'neggrad':
                    config['learning_rate'] /= 5
                
                config['model_name_or_path'] = get_full_model_name(b)
                config['dataset_name'] = d
                config['seed'] = s
                config['output_dir'] = f'checkpoint/unlearn/{out_dir}/{out_name}'
                config['hub_model_id'] = f'{b}-{m}-{dr}-{d}-{s}'


                with open(f'configs/unlearn/{out_dir}/{out_name}.json', 'w') as f:
                    json.dump(config, f, indent=4)