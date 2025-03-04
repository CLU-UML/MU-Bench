import os
import copy
import json


d = 'ddi'
backbones = ['biobert', 'pubmedbert-abstract', 'pubmedbert-fulltext']
seeds = [42, 87, 21, 100, 13]
del_ratio = [2.0, 4.0, 6.0, 8.0, 10.0]
methods = ['neggrad', 'random_label', 'bad_teaching', 'scrub', 'salun']


def get_full_model_name(m):
    if 'biobert' in m:
        m = 'dmis-lab/' + m + '-v1.1'

    elif 'pubmedbert-abstract' in m:
        m = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'

    elif 'pubmedbert-fulltext' in m:
        m = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'

    return m

template = {
    "do_train": True,
    "do_eval": True,
    "max_seq_length": 128,
    "num_train_epochs": 8,
    "logging_steps": 1000,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 256,
    "learning_rate": 5e-5,
    "warmup_steps": 0,
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
    "metric_name": "accuracy",
    "overwrite_output_dir": True,
    "dataloader_num_workers": 16,
    "push_to_hub": False,
}


# Train
s = 42
for b in backbones:
    config = copy.deepcopy(template)
    os.makedirs(f'configs/train', exist_ok=True)
    
    config['num_train_epochs'] = 20
    config['model_name_or_path'] = get_full_model_name(b)
    config['dataset_name'] = d
    config['seed'] = s
    config['output_dir'] = f'../../checkpoint/{d}/{b}'
    config['hub_model_id'] = f'{d}-{b}'

    with open(f'configs/train/{b}.json', 'w') as f:
        json.dump(config, f, indent=4)

# Unlearn
for b in backbones:
    for s in seeds:
        for dr in del_ratio:
            for m in methods:
                config = copy.deepcopy(template)
                out_dir = f'{b}/{m}/{dr}'
                out_name = f'{s}'
                os.makedirs(f'configs/unlearn/{out_dir}', exist_ok=True)

                config['unlearn_method'] = m
                config['del_ratio'] = dr
                if m == 'neggrad':
                    config['num_train_epochs'] = 6
                    config['learning_rate'] /= 5

                if m == 'bad_teaching':
                    config['learning_rate'] *= 10
                
                config['model_name_or_path'] = get_full_model_name(b)
                config['dataset_name'] = d
                config['seed'] = s
                config['output_dir'] = f'../../checkpoint/unlearn/{d}/{out_dir}/{out_name}'
                config['hub_model_id'] = f'{d}-{b}-{m}-{dr}-{s}'


                with open(f'configs/unlearn/{out_dir}/{out_name}.json', 'w') as f:
                    json.dump(config, f, indent=4)

# Unlearn for CL
for b in backbones:
    for s in seeds:
        for dr in del_ratio:
            for m in methods:
                config = copy.deepcopy(template)
                out_dir = f'{b}/{m}/{dr}'
                out_name = f'{s}'
                os.makedirs(f'configs/unlearn_cl/{out_dir}', exist_ok=True)

                config['unlearn_method'] = m
                config['del_ratio'] = dr
                if m == 'neggrad':
                    config['num_train_epochs'] = 6
                    config['learning_rate'] /= 5

                if m == 'bad_teaching':
                    config['learning_rate'] *= 10
                
                config['model_name_or_path'] = get_full_model_name(b)
                config['dataset_name'] = d
                config['seed'] = s
                config['use_cl'] = True
                config['output_dir'] = f'../../checkpoint/unlearn_cl/{d}/{out_dir}/{out_name}'
                config['hub_model_id'] = f'{d}-{b}-{m}-{dr}-{s}'


                with open(f'configs/unlearn_cl/{out_dir}/{out_name}.json', 'w') as f:
                    json.dump(config, f, indent=4)

# Unlearn SO
for b in backbones:
    for s in seeds:
        for dr in del_ratio:
            for m in methods:
                config = copy.deepcopy(template)
                out_dir = f'{b}/{m}/{dr}'
                out_name = f'{s}'
                os.makedirs(f'configs/unlearn_so/{out_dir}', exist_ok=True)

                config['unlearn_method'] = m
                config['del_ratio'] = dr
                if m == 'neggrad':
                    config['num_train_epochs'] = 6
                    config['learning_rate'] /= 5

                if m == 'bad_teaching':
                    config['learning_rate'] *= 10
                
                config['model_name_or_path'] = get_full_model_name(b)
                config['dataset_name'] = d
                config['seed'] = s
                config['use_so_info'] = True
                config['output_dir'] = f'../../checkpoint/unlearn_so/{d}/{out_dir}/{out_name}'
                config['hub_model_id'] = f'{d}-{b}-{m}-{dr}-{s}'


                with open(f'configs/unlearn_so/{out_dir}/{out_name}.json', 'w') as f:
                    json.dump(config, f, indent=4)
