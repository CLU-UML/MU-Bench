#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import logging
import os
import sys
import wandb
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

import transformers
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

sys.path.append(os.getcwd())
from m3u.args import UnlearningArguments
from m3u.trainer import UnlearningTrainer
from m3u.data.base import DeletionData, prepare_deletion_data


""" Fine-tuning a 🤗 Transformers model for image classification"""


model_map_rev = {
    'google/vit-base-patch16-224-in21k': 'vit-base-patch16-224',
    'google/vit-large-patch16-224-in21k': 'vit-large-patch16-224',
    'facebook/convnext-base-224': 'convnext-base-224',
    'facebook/convnext-base-224-22k': 'convnext-base-224-22k',
    'microsoft/resnet-18': 'resnet-18',
    'microsoft/resnet-34': 'resnet-34',
    'microsoft/resnet-50': 'resnet-50',
    'microsoft/swin-tiny-patch4-window7-224': 'swin-tiny',
    'microsoft/swin-base-patch4-window7-224': 'swin-base',
    'google/mobilenet_v2_1.0_224': 'mobilenet_v2',
}

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.28.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    image_column_name: str = field(
        default="image",
        metadata={"help": "The name of the dataset column containing the image data. Defaults to 'image'."},
    )
    label_column_name: str = field(
        default="label",
        metadata={"help": "The name of the dataset column containing the labels. Defaults to 'label'."},
    )

    def __post_init__(self):
        if self.dataset_name is None and (self.train_dir is None and self.validation_dir is None):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, UnlearningArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, unlearn_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, unlearn_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_image_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if unlearn_args.data_name is None:
        if data_args.dataset_config_name is not None:
            unlearn_args.data_name = data_args.dataset_name + '_' + data_args.dataset_config_name
        else:
            unlearn_args.data_name = data_args.dataset_name
        unlearn_args.random_seed = training_args.seed
        unlearn_args.backbone = model_map_rev[model_args.model_name_or_path]
    
    training_args.metric_for_best_model = 'unlearn_overall_' + training_args.metric_for_best_model

    training_args.learning_rate *= 2
    unlearn_args.lora_ratio = 50
    unlearn_args.use_lora = True
    from peft import LoraConfig, get_peft_model
    training_args.output_dir = training_args.output_dir.replace('checkpoint', f'lora_{unlearn_args.lora_ratio}_checkpoint')
    training_args.label_names = ['labels']

    # Wandb
    training_args.report_to = ['wandb']
    project = 'Unlearning Benchmark'
    group = unlearn_args.data_name + '-' + unlearn_args.backbone + f'-lora-{unlearn_args.lora_ratio}'
    name = unlearn_args.unlearn_method + '-' + str(unlearn_args.del_ratio) + '-' + str(unlearn_args.random_seed)
    run_id = '-'.join([unlearn_args.data_name, unlearn_args.backbone, f'lora-{unlearn_args.lora_ratio}', unlearn_args.unlearn_method, str(unlearn_args.del_ratio), str(unlearn_args.random_seed)])
    wandb.init(project=project, group=group, name=name, config=unlearn_args, id=run_id, resume='allow')

    # Log on each process the small summary:
    # logger.warning(
    #     f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    #     + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    # )
    # logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize our dataset and prepare it for the 'image-classification' task.
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            # task="image-classification",
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_dir is not None:
            data_files["train"] = os.path.join(data_args.train_dir, "**")
        if data_args.validation_dir is not None:
            data_files["validation"] = os.path.join(data_args.validation_dir, "**")
        raw_datasets = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            task="image-classification",
        )

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example[data_args.label_column_name] for example in examples])
        out = {"pixel_values": pixel_values, "labels": labels}

        if 'is_df' in examples[0]:
            out['is_df'] = torch.tensor([example['is_df'] for example in examples])

        return out


    # If we don't have a validation split, split off a percentage of train as validation.
    # data_args.train_val_split = None if "validation" in dataset.keys() else data_args.train_val_split
    # if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
    #     split = dataset["train"].train_test_split(data_args.train_val_split)
    #     dataset["train"] = split["train"]
    #     dataset["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    data_args.image_column_name = 'img'
    data_args.label_column_name = 'fine_label'
    labels = raw_datasets["train"].features[data_args.label_column_name].names
    label2id, id2label = {}, {}
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Load the accuracy metric from the datasets package
    metric = evaluate.load("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        # finetuning_task="image-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    ori_model = AutoModelForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

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
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch[data_args.image_column_name]
        ]

        if unlearn_args.unlearn_method in ['bad_teaching']:
            example_batch['is_df'] = list(example_batch['is_df'])
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [_val_transforms(pil_img.convert("RGB")) for pil_img in example_batch[data_args.image_column_name]]
        return example_batch

    ckpt_dir = f'./image/train/checkpoint/{model_map_rev[model_args.model_name_or_path]}/{data_args.dataset_name.split("/")[-1]}_42'
    if os.path.exists(os.path.join(ckpt_dir, 'pytorch_model.bin')):
        ckpt = torch.load(os.path.join(ckpt_dir, 'pytorch_model.bin'), map_location='cpu')
    elif os.path.exists(os.path.join(ckpt_dir, 'model.safetensors')):
        from safetensors.torch import load_file
        ckpt = load_file(os.path.join(ckpt_dir, 'model.safetensors'), device='cpu')
    ori_model.load_state_dict(ckpt, strict=False)


    # Unlearning data
    raw_datasets['ori_train'] = raw_datasets['train']
    raw_datasets['train'], raw_datasets['dr'], raw_datasets['df'], raw_datasets['df_for_train'] = prepare_deletion_data(unlearn_args, raw_datasets['ori_train'], data_args.label_column_name)
    # raw_datasets['train'] = DeletionData(unlearn_args, raw_datasets['ori_train'])
    # raw_datasets['train'] = Dataset.from_list(raw_datasets['train'])
    # raw_datasets['df'] = raw_datasets['train'].df_data
    # raw_datasets['dr'] = raw_datasets['train'].dr_data

    print('ssssssss', len(raw_datasets['train']))

    raw_datasets["train"].set_transform(train_transforms)
    raw_datasets["df_for_train"].set_transform(train_transforms)
    # raw_datasets["validation"].set_transform(val_transforms)
    raw_datasets["test"].set_transform(val_transforms)
    raw_datasets["df"].set_transform(val_transforms)
    # training_args.per_device_eval_batch_size = 8
    
    # Unlearning method
    if unlearn_args.unlearn_method == 'retrain':
        model = AutoModelForImageClassification.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    elif unlearn_args.unlearn_method in ['neggrad', 'fisher', 'random_label', 'bad_teaching', 'scrub', 'salul']:
        model = ori_model
        if unlearn_args.unlearn_method in ['bad_teaching']:
            training_args.remove_unused_columns = False

    from m3u.lora_config import lora_config_mapping
    model = get_peft_model(model, lora_config_mapping[(unlearn_args.backbone, unlearn_args.lora_ratio)])
    model.print_trainable_parameters()

    # training_args.num_train_epochs = 2
    # training_args.max_steps = 1
    # training_args.evaluation_strategy = 'steps'
    # training_args.eval_steps = 1
    # training_args.do_train = False

    # Initalize our trainer
    trainer = UnlearningTrainer(
        raw_datasets=raw_datasets,
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"] if training_args.do_train else None,
        eval_dataset=raw_datasets["test"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=image_processor,
        data_collator=collate_fn,
        unlearn_config=unlearn_args,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.unlearn(resume_from_checkpoint=checkpoint)
        # trainer.save_model()
        if train_result is not None:
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        # logger.info("*** Evaluate ***")
        # metrics = trainer.evaluate(eval_dataset=raw_datasets["validation"], metric_key_prefix='eval')
        # trainer.log_metrics("eval", metrics)
        # trainer.save_metrics("eval", metrics)

        logger.info("*** Test ***")
        metrics = trainer.evaluate(eval_dataset=raw_datasets["test"], metric_key_prefix='test')
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        logger.info("*** Dr ***")
        raw_datasets["dr"].set_transform(val_transforms)
        metrics = trainer.evaluate(eval_dataset=raw_datasets["dr"], metric_key_prefix='dr')
        trainer.log_metrics("dr", metrics)
        trainer.save_metrics("dr", metrics)

    # Write model card and (optionally) push to hub
    # kwargs = {
    #     "finetuned_from": model_args.model_name_or_path,
    #     "tasks": "image-classification",
    #     "dataset": data_args.dataset_name,
    #     "tags": ["image-classification", "vision"],
    # }
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()