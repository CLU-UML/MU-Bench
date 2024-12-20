from .args import UnlearningArguments
from .data.base import load_unlearn_data
from .utils import load_base_model

from .bad_teaching import BadTeachingTrainer
from .grad_ascent import GradAscentTrainer
from .random_labeling import RandomLabelingTrainer
from .scrub import SCRUBTrainer
from .salun import SalUnTrainer


def unlearn_trainer(trainer_name):
    trainer_methods = {
        "bad_teaching": BadTeachingTrainer,
        "grad_ascent": GradAscentTrainer,
        "neggrad": GradAscentTrainer,
        "random_labeling": RandomLabelingTrainer,
        "scrub": SCRUBTrainer,
        "salun": SalUnTrainer,
    }

    # Check if the trainer_name exists in the dictionary
    if trainer_name in trainer_methods:
        return trainer_methods[trainer_name]
    else:
        # Raise an error if the trainer_name is not recognized
        raise ValueError(f"Trainer '{trainer_name}' not found. Available trainers are: {', '.join(trainer_methods.keys())}")


## Mapping of base model names (short name to full name in HuggingFace)
model_map_rev = {
    # CIFAR100
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

    # IMDB
    'bert-base-uncased': 'bert-base',
    'FacebookAI/roberta-base': 'roberta-base',
    'distilbert/distilbert-base-uncased': 'distilbert-base',
    'google/electra-base-discriminator': 'electra-base',
    'microsoft/deberta-v3-base': 'deberta-base',
    'albert/albert-base-v2': 'albert-base-v2',
    'dmis-lab/biobert-v1.1': 'biobert',
    'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract': 'pubmedbert-abstract',
    'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext': 'pubmedbert-fulltext',

    # DDI2013


    # NLVR2


    # Speech Commands


}

model_map = {j: i for i, j in model_map_rev.items()}
