import os
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import datasets
from datasets import load_dataset, concatenate_datasets, interleave_datasets, DatasetDict
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
import mubench
from .tofu import *


def find_data_files(data_name):
    """
    Look for the data file in the following directories, in order:
    1. ../../ (two levels up)
    2. mubench/raw_data
    3. ~/.cache/mubench
    
    Returns the full path to the file if found, otherwise None.
    """
    package_folder = Path(mubench.__path__[0])
    search_dirs = [
        package_folder.parent.parent,           # ../../
        package_folder / "raw_data",            # mubench/raw_data
        Path(os.getenv('XDG_CACHE_HOME', Path.home() / ".cache")) / "mubench"  # .cache
    ]

    # Look for the file in the specified directories
    for directory in search_dirs:
        file_path = directory / filename
        if file_path.exists():
            return file_path

    # If file not found, run the download script
    download_script = package_folder / "script" / f"download_{data_name}.sh"
    if download_script.exists():
        try:
            print("File not found. Running download script...")
            subprocess.run(["bash", str(download_script)], check=True)
            print("Download script executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error running download script: {e}")
            return None
    else:
        print("Download script not found. Cannot download data.")
        return None

    # After downloading, check if the file exists in the package_folder/data
    downloaded_file_path = package_folder / "raw_data" / filename
    if downloaded_file_path.exists():
        return downloaded_file_path

    # If the file is still not found, return None
    print("Data file not found even after attempting download.")

    return None

def load_unlearn_data(unlearn_config, train_transforms=None, eval_transforms=None):
    # Dictionary to map dataset names to their respective load functions
    dataset_loaders = {
        "cifar10": load_cifar10,
        "cifar100": load_cifar100,
        "imdb": load_imdb,
        "ddi": load_ddi2013,
        "ddi2013": load_ddi2013,
        "nlvr2": load_nlvr2,
        "speech_commands": load_speech_commands,
        "ucf": load_ucf101,
        "ucf101": load_ucf101,
        "samsum": load_samsum,
        "celeb_profile": load_celeb_profile,
        "tiny_imagenet": load_tiny_imagenet
    }

    if 'tofu' not in unlearn_config.data_name:
        # Check if the data_name exists in the dictionary
        if unlearn_config.data_name not in dataset_loaders:
            raise ValueError(f"Dataset '{unlearn_config.data_name}' not found. Available datasets are: {', '.join(dataset_loaders.keys())}")

        # Load data, train, valid, test
        raw_datasets = dataset_loaders[unlearn_config.data_name]()

        # Prepare Df, Dr, method-specific training set, and unlearning eval set
        raw_datasets = prepare_unlearning_data(unlearn_config, raw_datasets)

        if train_transforms is not None and eval_transforms is not None:
            for split in raw_datasets.keys():
                if 'train' in split:
                    print(f'Set transform for {split} as {train_transforms}')
                    raw_datasets[split].set_transform(train_transforms)
                else:
                    print(f'Set transform for {split} as {eval_transforms}')
                    raw_datasets[split].set_transform(eval_transforms)

    else:   # TOFU
        raw_datasets = {}
        raw_datasets['df'] = load_dataset("locuslab/TOFU", f'forget{int(float(unlearn_config.del_ratio)):>02d}')["train"]
        raw_datasets['dr'] = load_dataset("locuslab/TOFU", f'retain{int(float(100-unlearn_config.del_ratio)):>02d}')["train"]

        raw_datasets = DatasetDict(raw_datasets)

        tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-1_5')

        def process(examples):
            out = [convert_raw_data_to_model_format(tokenizer, 500, i, j, 'phi-1.5') 
                for i, j in zip(examples['question'], examples['answer'])]
            output = {
                'input_ids': [i[0] for i in out],
                'attention_mask': [i[2] for i in out],
                'label_ids': [i[1] for i in out],
            }
            return output

        raw_datasets = raw_datasets.map(process, batched=True, remove_columns=['question', 'answer'])
        raw_datasets = method_specific_transformation(unlearn_config, raw_datasets, raw_datasets['df'], raw_datasets['dr'])

    return raw_datasets

def load_cifar10():
    raw_datasets = load_dataset('cifar10')
    raw_datasets = raw_datasets.rename_column('img', 'image')

    return raw_datasets

def load_cifar100():
    raw_datasets = load_dataset('cifar100')
    raw_datasets = raw_datasets.rename_column('fine_label', 'label')
    raw_datasets = raw_datasets.rename_column('img', 'image')

    return raw_datasets

def load_imdb():
    raw_datasets = load_dataset('imdb')
    ood = load_dataset('rotten_tomatoes')
    raw_datasets['ood'] = concatenate_datasets([ood['train'], ood['validation'], ood['test']])

    return raw_datasets

def load_ddi2013():
    train = pd.read_csv(f'{mubench.__path__[0]}/raw_data/ddi/train.tsv', sep='\t', names=['none', 'text', 'label']).drop('none', axis=1)
    dev = pd.read_csv(f'{mubench.__path__[0]}/raw_data/ddi/dev.tsv', sep='\t', names=['none', 'text', 'label']).drop('none', axis=1)
    test = pd.read_csv(f'{mubench.__path__[0]}/raw_data/ddi/test.tsv', sep='\t', names=['none', 'text', 'label']).drop('none', axis=1)

    label_mapping = {'DDI-false': 0, 'DDI-mechanism': 1, 'DDI-advise': 2, 'DDI-effect': 3, 'DDI-int': 4}
    train.label = train.label.apply(label_mapping.get)
    dev.label = dev.label.apply(label_mapping.get)
    test.label = test.label.apply(label_mapping.get)

    raw_datasets = DatasetDict({
        'train': HFDataset.from_pandas(train),
        'validation': HFDataset.from_pandas(dev),
        'test': HFDataset.from_pandas(test),
    })

    return raw_datasets

def load_nlvr2():
    raw_datasets = load_dataset(
        'jialicheng/nlvr2',
        data_files={'train': 'train.json', 'validation': 'dev.json', 'test': 'test.json', 'ood': 'test2.json'}
    )

    def mapping(examples):
        label2id = {'False': 0, 'True': 1}
        id2label = {0: 'False', 1: 'True'}
        examples['label'] = [label2id[i] for i in examples['label']]
    
        return examples

    raw_datasets = raw_datasets.map(mapping, batched=True)

    return raw_datasets

def load_speech_commands():
    return load_dataset('superb', 'ks')

def load_ucf101():
    pass
def load_samsum():
    # ("dialogue", "summary")
    return load_dataset('samsum')

def load_bioceleb():
    pass
def load_celeb_profile():
    pass
def load_tiny_imagenet():
    pass

def load_tofu():
    return load_dataset("locuslab/TOFU", "full")

def _corrupt_label(df_data, dr_data, label_col, is_generative_task=False, seed=None):
    rng = np.random.default_rng(seed)
    original_label = df_data[label_col]
    dr_label = dr_data[label_col]

    if is_generative_task:  # Choose a random label from Dr
        random_label_idx = rng.choice(np.arange(len(dr_label)), size=len(original_label), replace=False)
        corrupted_label = [dr_label[i] for i in random_label_idx]
    
    else:                   # Increament label by 1. If use the method above, corrupted label may be the same as original label.
        num_labels = len(set(dr_label))
        corrupted_label = [(i + 1) % num_labels for i in original_label]

    assert all([i != j for i, j in zip(corrupted_label, original_label)])
    # assert all([i != j for i, j in zip(df_data[label_col], df_data['orig_label'])])

    df_data = df_data.remove_columns([label_col])
    # df_data = df_data.rename_column(label_col, 'orig_label')
    df_data = df_data.add_column(label_col, corrupted_label)

    return df_data

def prepare_df_dr(unlearn_config, train_dataset):
    
    # Create binary mask for Df, Dr
    del_idx = np.loadtxt(f'{mubench.__path__[0]}/df/{unlearn_config.data_name}.txt', dtype=int)
    del_idx = del_idx[:int(unlearn_config.del_ratio / 10 * len(del_idx))]
    df_mask = np.zeros(train_dataset.shape[0], dtype=bool)
    df_mask[np.array(del_idx)] = True
    dr_mask = ~df_mask
    
    all_idx = np.arange(train_dataset.shape[0])
    df_data = datasets.Dataset.from_dict(train_dataset[all_idx[df_mask]])
    dr_data = datasets.Dataset.from_dict(train_dataset[all_idx[dr_mask]])

    assert dr_data.shape[0] < train_dataset.shape[0]
    assert dr_data.shape[0] + df_data.shape[0] == train_dataset.shape[0]

    return df_data, dr_data

def method_specific_transformation(unlearn_config, raw_datasets, df_data, dr_data):
    if unlearn_config.unlearn_method in ['retrain', 'fisher', 'l-codec', 'scrub']:
        raw_datasets['train'] = copy.deepcopy(dr_data)

    elif unlearn_config.unlearn_method in ['neggrad']:
        raw_datasets['train'] = copy.deepcopy(df_data)

    elif unlearn_config.unlearn_method in ['random_label', 'salun']:
        df_train = copy.deepcopy(df_data)

        if unlearn_config.unlearn_method == 'salun':
            raw_datasets['df_train'] = df_data

        dr_train = copy.deepcopy(dr_data)
        df_corrupted = _corrupt_label(df_train, dr_train, label_col, is_generative_task, unlearn_config.random_seed)
        
        raw_datasets['train'] = concatenate_datasets([df_corrupted, dr_data])


    elif unlearn_config.unlearn_method in ['bad_teaching']:     # We use "is_df" to denote the membership of samples
        df_train = copy.deepcopy(df_data)

        all_idx = np.arange(dr_data.shape[0])
        rng = np.random.default_rng(unlearn_config.random_seed)
        sel_idx = np.random.choice(all_idx, size=df_data.shape[0], replace=False)

        dr_subset = datasets.Dataset.from_dict(raw_datasets['train'][sel_idx])
        dr_subset = dr_subset.add_column('is_df', [0,] * df_data.shape[0])
        df_data = df_data.add_column('is_df', [1,] * df_data.shape[0])

        raw_datasets['train'] = interleave_datasets([dr_subset, df_data], stopping_strategy='all_exhausted')
    
    return raw_datasets

def prepare_unlearning_data(unlearn_config, raw_datasets, label_col='label', is_generative_task=False, do_method_specific_transformation=True):
    df_data, dr_data = prepare_df_dr(unlearn_config, raw_datasets['train'])
    
    raw_datasets['orig_train'] = raw_datasets['train']
    raw_datasets['df'] = df_data
    raw_datasets['dr'] = dr_data

    # if unlearn_config.unlearn_method in ['random_label', 'salul']:
    #     num_labels = len(set(raw_datasets[label_col]))#.names)
    #     original_label = df_data[label_col]

    #     if is_generative_task:
    #         dr_label = dr_data[label_col]
    #         corrupted_label = [dr_label[i] for i in range(len(original_label))]
        
    #     else:
    #         num_labels = len(set(raw_datasets[label_col]))#.names)
    #         corrupted_label = [(i-1) % num_labels for i in original_label]

    if not do_method_specific_transformation:
        return raw_datasets

    # Do method-specific data transformation
    if unlearn_config.unlearn_method in ['retrain', 'fisher', 'l-codec', 'scrub']:
        raw_datasets['train'] = copy.deepcopy(dr_data)

    elif unlearn_config.unlearn_method in ['neggrad']:
        raw_datasets['train'] = copy.deepcopy(df_data)

    elif unlearn_config.unlearn_method in ['random_label', 'salun']:
        df_train = copy.deepcopy(df_data)

        if unlearn_config.unlearn_method == 'salun':
            raw_datasets['df_train'] = df_data

        dr_train = copy.deepcopy(dr_data)
        df_corrupted = _corrupt_label(df_train, dr_train, label_col, is_generative_task, unlearn_config.random_seed)
        
        raw_datasets['train'] = concatenate_datasets([df_corrupted, dr_data])


    elif unlearn_config.unlearn_method in ['bad_teaching']:     # We use "is_df" to denote the membership of samples
        df_train = copy.deepcopy(df_data)

        all_idx = np.arange(dr_data.shape[0])
        rng = np.random.default_rng(unlearn_config.random_seed)
        sel_idx = np.random.choice(all_idx, size=df_data.shape[0], replace=False)

        dr_subset = datasets.Dataset.from_dict(raw_datasets['train'][sel_idx])
        dr_subset = dr_subset.add_column('is_df', [0,] * df_data.shape[0])
        df_data = df_data.add_column('is_df', [1,] * df_data.shape[0])

        raw_datasets['train'] = interleave_datasets([dr_subset, df_data], stopping_strategy='all_exhausted')

        # Another implementation is to pass Df and Dr for each sample
        # times = dr_data.shape[0] // df_data.shape[0]
        # remainder = dr_data.shape[0] % df_data.shape[0]

        # # Repeat Df to be of same size as Dr
        # repeated_df = [df_data,] * (times+1)
        # repeated_df = concatenate_datasets(repeated_df)
        # repeated_df = HFDataset.from_dict(repeated_df[:dr_data.shape[0]])

        # # Add prefix
        # col = dr_data.column_names
        # dr_data = dr_data.rename_columns({i: f'dr_{i}' for i in col})
        # repeated_df = repeated_df.rename_columns({i: f'df_{i}' for i in col})

        # interleave_data = concatenate_datasets([dr_data, repeated_df], axis=1)

    return raw_datasets

def prepare_unlearning_data_video(unlearn_config, ori_train_data, label_col='label', is_generative_task=False, method_specific_transformation=True):
    import pandas as pd
    df_mask = np.loadtxt(f'{mubench.__path__[0]}/df/{unlearn_config.data_name}/{unlearn_config.del_ratio}.txt', dtype=bool)
    dr_mask = ~df_mask

    data = copy.deepcopy(ori_train_data)
    df_data = copy.deepcopy(ori_train_data)
    dr_data = copy.deepcopy(ori_train_data)
    df_for_train = copy.deepcopy(ori_train_data)
    dr_for_eval = copy.deepcopy(ori_train_data)
    
    ori_train_data_path = pd.DataFrame(ori_train_data._paths_and_labels, columns=['path', 'label'])
    df_path = pd.DataFrame(ori_train_data._paths_and_labels, columns=['path', 'label'])[df_mask]
    dr_path = pd.DataFrame(ori_train_data._paths_and_labels, columns=['path', 'label'])[dr_mask]
    df_for_train_path = pd.DataFrame(ori_train_data._paths_and_labels, columns=['path', 'label'])[df_mask]
    dr_for_eval_path = pd.DataFrame(ori_train_data._paths_and_labels, columns=['path', 'label'])[dr_mask][:df_mask.sum()]

    df_data._paths_and_labels = df_path.values.tolist()
    dr_data._paths_and_labels = dr_path.values.tolist()
    df_for_train._paths_and_labels = df_for_train_path.values.tolist()
    dr_for_eval._paths_and_labels = dr_for_eval_path.values.tolist()


    if unlearn_config.unlearn_method in ['random_label', 'salul']:
        num_labels = len(set(ori_train_data_path[label_col]))#.names)
        original_label = df_path[label_col]
        corrupted_label = [(i-1) % num_labels for i in original_label]

        corrupted_df_data = copy.deepcopy(df_data)
        corrupted_df_data_path = pd.DataFrame(corrupted_df_data._paths_and_labels, columns=['path', 'label'])
        corrupted_df_data_path['label'] = corrupted_label

        assert all([i != j for i, j in zip(corrupted_df_data_path['label'], df_path['label'])])
        concat_path = pd.concat([corrupted_df_data_path, dr_path], ignore_index=True).values.tolist()

        data._paths_and_labels = concat_path

    elif unlearn_config.unlearn_method in ['retrain', 'fisher', 'l-codec', 'scrub']:
        data = dr_data

    elif unlearn_config.unlearn_method in ['neggrad']:
        data = copy.deepcopy(df_data)

    elif unlearn_config.unlearn_method in ['bad_teaching']:
        # times = dr_data.shape[0] // df_data.shape[0]
        # remainder = dr_data.shape[0] % df_data.shape[0]

        # # Repeat Df to be of same size as Dr
        # repeated_df = [df_data,] * (times+1)
        # repeated_df = concatenate_datasets(repeated_df)
        # repeated_df = HFDataset.from_dict(repeated_df[:dr_data.shape[0]])

        # # Add prefix
        # col = dr_data.column_names
        # dr_data = dr_data.rename_columns({i: f'dr_{i}' for i in col})
        # repeated_df = repeated_df.rename_columns({i: f'df_{i}' for i in col})

        # interleave_data = concatenate_datasets([dr_data, repeated_df], axis=1)
        all_idx = np.arange(len(dr_data))
        sel_idx = np.random.choice(all_idx, size=len(df_data), replace=False)
        dr_subset = copy.deepcopy(dr_data)
        dr_subset._paths_and_labels = dr_subset._paths_and_labels[:len(df_path)]
        data = dr_subset
        # df_data = df_data.add_column('is_df', [1,] * df_data.shape[0])
        # dr_subset = dr_subset.add_column('is_df', [0,] * dr_subset.shape[0])
        # data = interleave_datasets([dr_subset, df_data], stopping_strategy='all_exhausted')

    # elif unlearn_config.unlearn_method in ['random_label']

    if unlearn_config.unlearn_method == 'salul':
        df_for_train = corrupted_df_data
    else:
        df_for_train = copy.deepcopy(df_data)

    # return HFDataset(data)
    return data, dr_data, df_data, df_for_train, dr_for_eval


class DeletionData(Dataset):
    def __init__(self, unlearn_config, ori_train_data, transform=None):
        self.unlearn_config = unlearn_config
        self.ori_train_data = ori_train_data
        self.df_mask = np.loadtxt(f'{mubench.__path__[0]}/df/text/{unlearn_config.data_name}/{unlearn_config.del_ratio}.txt', dtype=bool)
        self.dr_mask = ~self.df_mask

        self.df_data, self.dr_data = self.prepare_df_dr()
        
        if self.unlearn_config.unlearn_method in ['random_label']:
            num_labels = len(set(self.ori_train_data['label']))#.names)
            original_label = self.df_data['label']
            corrupted_label = [num_labels - i - 1 for i in original_label]
            corrupted_df_data = copy.deepcopy(self.df_data)
            corrupted_df_data = corrupted_df_data.rename_column('label', 'ori_label')
            corrupted_df_data = corrupted_df_data.add_column('label', corrupted_label)

            assert all([i != j for i, j in zip(corrupted_df_data['label'], self.df_data['label'])])
            self.concat_data = concatenate_datasets([corrupted_df_data, self.dr_data])
        # self.subset_dr_data = self.dr_data
        # self.resample_dr()

        # # These unlearning methods use the whole Dr as training set
        # if self.unlearn_config.unlearn_method in ['retrain', 'fisher', 'l-codec']:
        #     assert self.subset_dr_data.shape[0] == self.dr_data.shape[0]

        self.transform = transform

    def resample_dr(self):
        '''Sample a subset from Dr to have same size of Df, for some unlearning methods. 
            This is called every epoch to cover a wider range of Dr, without relying on the same sample
        '''

        # These methods iterate through Df, Dr simuteneously during training.
        if self.unlearn_config.unlearn_method in ['bad_teaching']:
            dr_size = self.dr_data.shape[0]
            df_size = self.df_data.shape[0]
            sel_idx = np.random.choice(np.arange(dr_size), size=df_size, replace=False)
            
            self.subset_dr_data = self.dr_data.select(sel_idx)#.reset_index(drop=True)

    def __len__(self):
        if self.unlearn_config.unlearn_method in ['retrain', 'fisher', 'l-codec', 'bad_teaching']:
            return self.dr_data.shape[0]
        elif self.unlearn_config.unlearn_method in ['random_label']:
            return self.concat_data.shape[0]
        else:
            return self.df_data.shape[0]

    def prepare_df_dr(self):
        all_idx = np.arange(self.ori_train_data.shape[0])
        df_data = datasets.Dataset.from_dict(self.ori_train_data[all_idx[self.df_mask]])
        dr_data = datasets.Dataset.from_dict(self.ori_train_data[all_idx[self.dr_mask]])

        assert dr_data.shape[0] < self.ori_train_data.shape[0]
        assert dr_data.shape[0] + df_data.shape[0] == self.ori_train_data.shape[0]

        return df_data, dr_data

    def __getitem__(self, idx):
        # dr = self.subset_dr_data[idx]
        # df = self.df_data[idx]# % self.df_data.shape[0]]  # Df is smaller than Dr. idx may be out of bound

        if self.unlearn_config.unlearn_method in ['random_label']:
            return self.concat_data[idx]

        dr = self.dr_data[idx]
        df = self.df_data[idx % self.df_data.shape[0]]  # Df is smaller than Dr. idx may be out of bound

        # data = {'dr_input': dr, 'df_input': df}
        dfdr = {}
        for k, v in df.items():
            if k == 'label':
                k = 'labels'
            dfdr['df_'+k] = v
        for k, v in dr.items():
            if k == 'label':
                k = 'labels'
            dfdr['dr_'+k] = v

        if self.unlearn_config.unlearn_method in ['neggrad']:
            return df

        elif self.unlearn_config.unlearn_method in ['bad_teaching']:
            if self.transform is not None:
                return self.transform(dfdr)
            else:
                return dfdr

        return dr


def prepare_dr_data(dataset_train_ori, cfg, data_type):
    with open(f'Df/{data_type}/image-{cfg.run_cfg.seed}.txt', 'r') as f:
        df_ids = f.readlines()
    df_ids = [i.strip() for i in df_ids]
    df_ids = df_ids[:cfg.run_cfg.df_size]
    df_ids_set = set(df_ids)

    dataset = copy.deepcopy(dataset_train_ori)

    if cfg.run_cfg.task == 'retrieval':
        num_image_before_removal = len(set([i['image'] for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if i['image'] not in df_ids_set]
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    elif cfg.run_cfg.task == 'vqa':
        num_image_before_removal = len(set([i['image'] for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if i['image'] not in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    elif cfg.model_cfg.model_type == 'nlvr':
        num_image_before_removal = len(set([str(tuple(i['images'])) for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if str(tuple(i['images'])) not in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([str(tuple(i['images'])) for i in dataset.annotation]))

    elif cfg.model_cfg.model_type == 've':
        num_image_before_removal = len(set([i['image'] for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if i['image'] not in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    # assert num_image_before_removal == num_image_after_removal + cfg.run_cfg.df_size

    return dataset

def prepare_df_data(dataset_train_ori, cfg, data_type):
    with open(f'Df/{data_type}/image-{cfg.run_cfg.seed}.txt', 'r') as f:
        df_ids = f.readlines()
    df_ids = [i.strip() for i in df_ids]
    df_ids = df_ids[:cfg.run_cfg.df_size]
    df_ids_set = set(df_ids)

    dataset = copy.deepcopy(dataset_train_ori)
    
    if cfg.run_cfg.task == 'retrieval':
        dataset.annotation = [i for i in dataset.annotation if i['image'] in df_ids_set]
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    elif cfg.run_cfg.task == 'vqa':
        dataset.annotation = [i for i in dataset.annotation if i['image'] in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    elif cfg.model_cfg.model_type == 'nlvr':
        num_image_before_removal = len(set([str(tuple(i['images'])) for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if str(tuple(i['images'])) in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([str(tuple(i['images'])) for i in dataset.annotation]))

    elif cfg.model_cfg.model_type == 've':
        num_image_before_removal = len(set([i['image'] for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if i['image'] in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    # assert num_image_after_removal == cfg.run_cfg.df_size, f"{num_image_after_removal}, {cfg.run_cfg.df_size}"

    return dataset


def prepare_df_data_for_test(dataset_train_ori, dataset_test_ori, cfg, data_type):
    with open(f'Df/{data_type}/image-{cfg.run_cfg.seed}.txt', 'r') as f:
        df_ids = f.readlines()
    df_ids = [i.strip() for i in df_ids]
    df_ids = df_ids[:cfg.run_cfg.df_size]
    df_ids_set = set(df_ids)


    if cfg.run_cfg.task == 'retrieval':
        # Retrieval train and test data are different. We want to use retrieval test data for Df. So copy the ori test data
        df_for_test = copy.deepcopy(dataset_test_ori)

        annotation = [i for i in dataset_train_ori.annotation if i['image'] in df_ids_set]
        num_image_after_removal = len(set([i['image'] for i in annotation]))

        # Convert to grouped format for init of RetrievalEvalDataset
        test_anno = pd.DataFrame(annotation).sort_values(by='image')
        test_anno = test_anno.groupby(['image'])['caption'].apply(list).reset_index()
        test_anno = test_anno.to_dict(orient='records')
        df_for_test.annotation = test_anno      # For __len__ method

        # init of RetrievalEvalDataset
        text = []
        image = []
        txt2img = {}
        img2txt = {}
        text_processor = df_for_test.text_processor

        txt_id = 0
        for img_id, ann in enumerate(test_anno):
            image.append(ann["image"])
            img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                text.append(text_processor(caption))
                img2txt[img_id].append(txt_id)
                txt2img[txt_id] = img_id
                txt_id += 1

        df_for_test.text = text
        df_for_test.image = image
        df_for_test.txt2img = txt2img
        df_for_test.img2txt = img2txt

    elif cfg.run_cfg.task == 'vqa':
        # breakpoint()
        # Retrieval train and test data are same. To use VQA test data for Df, copy the ori train data
        df_for_test = copy.deepcopy(dataset_train_ori)

        df_for_test.annotation = [i for i in df_for_test.annotation if i['image'] in df_ids_set]
        df_for_test._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in df_for_test.annotation]))
        # breakpoint()

    # elif cfg.run_cfg.task == 'multimodal_classification':
    #     breakpoint()
    #     df_for_test = copy.deepcopy(dataset_test_ori)

    #     df_for_test.annotation = [i for i in dataset_train_ori.annotation if i['image'] in df_ids_set]
    #     df_for_test._add_instance_ids()
    #     num_image_after_removal = len(set([i['image'] for i in df_for_test.annotation]))
    #     breakpoint()

    # NLVR train and test data are different. To use NLVR test data for Df, copy the ori test data
    elif cfg.model_cfg.model_type == 'nlvr':
        df_for_test = copy.deepcopy(dataset_test_ori)
        df_for_test.annotation = copy.deepcopy(dataset_train_ori.annotation)

        df_for_test.annotation = [i for i in df_for_test.annotation if str(tuple(i['images'])) in df_ids_set]
        df_for_test._add_instance_ids()
        num_image_after_removal = len(set([str(tuple(i['images'])) for i in df_for_test.annotation]))

    elif cfg.model_cfg.model_type in ['base', 've']:
        df_for_test = copy.deepcopy(dataset_test_ori)
        df_for_test.annotation = copy.deepcopy(dataset_train_ori.annotation)

        df_for_test.annotation = [i for i in df_for_test.annotation if i['image'] in df_ids_set]
        df_for_test._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in df_for_test.annotation]))

    # assert num_image_after_removal == cfg.run_cfg.df_size, f"{num_image_after_removal}, {cfg.run_cfg.df_size}"

    return df_for_test

def prepare_dr_data_for_test(dataset_train_ori, dataset_test_ori, cfg, data_type, sample_size=None):
    with open(f'Df/{data_type}/image-{cfg.run_cfg.seed}.txt', 'r') as f:
        df_ids = f.readlines()
    df_ids = [i.strip() for i in df_ids]
    df_ids = df_ids[:cfg.run_cfg.df_size]
    df_ids_set = set(df_ids)


    if cfg.run_cfg.task == 'retrieval':
        num_image_before_removal = len(set([i['image'] for i in dataset_train_ori.annotation]))

        # Retrieval train and test data are different. We want to use retrieval test data for Df. So copy the ori test data
        dr_for_test = copy.deepcopy(dataset_test_ori)

        annotation = [i for i in dataset_train_ori.annotation if i['image'] not in df_ids_set]
        num_image_after_removal = len(set([i['image'] for i in annotation]))

        # Convert to grouped format for init of RetrievalEvalDataset
        test_anno = pd.DataFrame(annotation).sort_values(by='image')
        test_anno = test_anno.groupby(['image'])['caption'].apply(list).reset_index()
        test_anno = test_anno.to_dict(orient='records')
        dr_for_test.annotation = test_anno      # For __len__ method

        # init of RetrievalEvalDataset
        text = []
        image = []
        txt2img = {}
        img2txt = {}
        text_processor = dr_for_test.text_processor

        txt_id = 0
        for img_id, ann in enumerate(test_anno):
            image.append(ann["image"])
            img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                text.append(text_processor(caption))
                img2txt[img_id].append(txt_id)
                txt2img[txt_id] = img_id
                txt_id += 1

        dr_for_test.text = text
        dr_for_test.image = image
        dr_for_test.txt2img = txt2img
        dr_for_test.img2txt = img2txt

    elif cfg.run_cfg.task == 'vqa':
        # breakpoint()
        # Retrieval train and test data are same. To use VQA test data for Df, copy the ori train data
        dr_for_test = copy.deepcopy(dataset_train_ori)

        dr_for_test.annotation = [i for i in dr_for_test.annotation if i['image'] not in df_ids_set]
        dr_for_test._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dr_for_test.annotation]))
        # breakpoint()

    # elif cfg.run_cfg.task == 'multimodal_classification':
    #     breakpoint()
    #     dr_for_test = copy.deepcopy(dataset_test_ori)

    #     dr_for_test.annotation = [i for i in dataset_train_ori.annotation if i['image'] in df_ids_set]
    #     dr_for_test._add_instance_ids()
    #     num_image_after_removal = len(set([i['image'] for i in dr_for_test.annotation]))
    #     breakpoint()

    # NLVR train and test data are different. To use NLVR test data for Df, copy the ori test data
    elif cfg.model_cfg.model_type == 'nlvr':
        dr_for_test = copy.deepcopy(dataset_test_ori)
        dr_for_test.annotation = copy.deepcopy(dataset_train_ori.annotation)

        num_image_before_removal = len(set([str(tuple(i['images'])) for i in dr_for_test.annotation]))
        dr_for_test.annotation = [i for i in dr_for_test.annotation if str(tuple(i['images'])) not in df_ids_set]

        if sample_size is not None:
            anno_id = np.arange(len(dr_for_test.annotation))
            indices = np.random.choice(anno_id, sample_size, replace=False)
            dr_for_test.annotation = [dr_for_test.annotation[i] for i in indices]

        dr_for_test._add_instance_ids()
        num_image_after_removal = len(set([str(tuple(i['images'])) for i in dr_for_test.annotation]))

    elif cfg.model_cfg.model_type in ['base', 've']:
        dr_for_test = copy.deepcopy(dataset_test_ori)
        dr_for_test.annotation = copy.deepcopy(dataset_train_ori.annotation)

        num_image_before_removal = len(set([i['image'] for i in dr_for_test.annotation]))
        dr_for_test.annotation = [i for i in dr_for_test.annotation if i['image'] not in df_ids_set]
        dr_for_test._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dr_for_test.annotation]))

    # assert num_image_before_removal == num_image_after_removal + cfg.run_cfg.df_size

    return dr_for_test