import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import (
    TrainingArguments,
    AutoImageProcessor,
    AutoTokenizer,
    AutoProcessor,
    default_data_collator
)
import mubench


all_tokenizers = {
    "cifar10": AutoImageProcessor,
    "cifar100": AutoImageProcessor,
    "imdb": AutoTokenizer,
    "ddi": AutoTokenizer,
    "ddi2013": AutoTokenizer,
    "speech_commands": AutoTokenizer,
    "ucf101": AutoTokenizer,
    "samsum": AutoTokenizer,
    "celeb_profile": AutoTokenizer,
    "tiny_imagenet": AutoImageProcessor,
    "tofu": AutoTokenizer,
}

def tokenize_data(unlearn_config, raw_datasets):
    dataset_tokenizers = {
        "cifar10": preprocess_for_image_classification,
        "cifar100": preprocess_for_image_classification,
        "imdb": preprocess_for_text_classification,
        "ddi": preprocess_for_text_classification,
        "ddi2013": preprocess_for_text_classification,
    }

    # Check if the data_name exists in the dictionary
    if unlearn_config.data_name not in dataset_tokenizers:
        raise ValueError(f"Dataset '{unlearn_config.data_name}' not found. Available datasets are: {', '.join(dataset_tokenizers.keys())}")

    return dataset_tokenizers[unlearn_config.data_name](unlearn_config, raw_datasets)


def preprocess_for_image_classification(unlearn_config, raw_datasets):
    image_processor = AutoImageProcessor.from_pretrained(mubench.model_map[unlearn_config.backbone])

    # Define torchvision transforms to be applied to each image.
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    _train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch['image']
        ]

        if 'is_df' in example_batch:
            example_batch['is_df'] = list(example_batch['is_df'])
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [_val_transforms(pil_img.convert("RGB")) for pil_img in example_batch['image']]
        return example_batch

    for split in raw_datasets.keys():
        if 'train' in split:
            print(f'Set transform for {split} as {train_transforms}')
            raw_datasets[split].set_transform(train_transforms)
        else:
            print(f'Set transform for {split} as {val_transforms}')
            raw_datasets[split].set_transform(val_transforms)

    return raw_datasets


def preprocess_for_text_classification(unlearn_config, raw_datasets):
    tokenizer = AutoTokenizer.from_pretrained(mubench.model_map[unlearn_config.backbone])

    def basic_tokenization(examples):
        results = tokenizer(examples["text"], padding='max_length', max_length=128, truncation=True)
        return results

    training_args = TrainingArguments(output_dir='tmp')
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            basic_tokenization,
            batched=True,
            desc=f"Tokenization",
            load_from_cache_file=True,
        )
    del training_args

    return raw_datasets


def collate_fn_image_classification(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example['label'] for example in examples])
    out = {"pixel_values": pixel_values, "labels": labels}

    if 'is_df' in examples[0]:
        out['is_df'] = torch.tensor([example['is_df'] for example in examples])

    return out

all_dataset_collators = {
    'cifar10': collate_fn_image_classification,
    'cifar100': collate_fn_image_classification,
    'imdb': default_data_collator,
    'ddi': default_data_collator,
    'ddi2013': default_data_collator,
}
