import os
import copy
import json


d = 'imdb'
backbones = ['bert-base', 'roberta-base', 'distilbert-base', 'electra-base', 'deberta-base', 'albert-base-v2']
seeds = [42, 87, 21, 100, 13]
del_ratio = [2.0, 4.0, 6.0, 8.0, 10.0]
methods = ['neggrad', 'random_label', 'bad_teaching', 'scrub', 'salun']


def get_full_model_name(m):
    if m.startswith('bert-'):
        m = m + '-uncased'

    elif 'roberta' in m:
        m = 'FacebookAI/' + m

    elif 'distilbert-' in m:
        m = 'distilbert/' + m + '-uncased'

    elif 'electra-' in m:
        m = 'google/' + m + '-discriminator'

    elif 'deberta-' in m:
        if 'base' in m:
            m = 'microsoft/deberta-v3-base'
    
    elif 'albert-' in m:
        m = 'albert/' + m
    
    elif 'biobert' in m:
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
    "num_train_epochs": 10,
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
    config['seed'] = s
    config['output_dir'] = f'../../checkpoint/imdb/{b}'
    config['hub_model_id'] = f'imdb-{b}'

    with open(f'configs/train/{b}.json', 'w') as f:
        json.dump(config, f, indent=4)

# Unlearn
for b in backbones:
    for s in seeds:
        for dr in del_ratio:
            for m in methods:
                config = copy.deepcopy(template)
                out_dir = f'{b}/{m}/{dr}'
                out_name = f'imdb_{s}'
                os.makedirs(f'configs/unlearn/{out_dir}', exist_ok=True)

                config['unlearn_method'] = m
                config['del_ratio'] = dr
                if m == 'neggrad':
                    config['learning_rate'] /= 5
                
                config['model_name_or_path'] = get_full_model_name(b)
                config['seed'] = s
                config['output_dir'] = f'checkpoint/unlearn/imdb/{out_dir}/{out_name}'
                config['hub_model_id'] = f'{b}-{m}-{dr}-imdb-{s}'


                with open(f'configs/unlearn/{out_dir}/{out_name}.json', 'w') as f:
                    json.dump(config, f, indent=4)


backbones = ['biobert', 'pubmedbert-abstract', 'pubmedbert-fulltext']
