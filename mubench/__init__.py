from .args import UnlearningArguments
from .data.base import load_unlearn_data
from .utils import load_base_model
from .trainer import get_trainer


## Mapping of base model names (short name to full name in HuggingFace)
model_map_rev = {
    # CIFAR100
    'google/vit-base-patch16-224-in21k': 'vit-base',
    'google/vit-large-patch16-224-in21k': 'vit-large',
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

    # DDI2013
    'dmis-lab/biobert-v1.1': 'biobert',
    'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract': 'pubmedbert-abstract',
    'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext': 'pubmedbert-fulltext',


    # NLVR2


    # Speech Commands
    'facebook/wav2vec2-base': 'wav2vec2-base',
    'facebook/wav2vec2-large': 'wav2vec2-large',
    'facebook/hubert-base-ls960': 'hubert-base',
    'facebook/hubert-large-ll60k': 'hubert-large',
    'facebook/hubert-xlarge-ll60k': 'hubert-xlarge',
    'openai/whisper-tiny': 'whisper-tiny',
    'openai/whisper-base': 'whisper-base',

}

model_map = {j: i for i, j in model_map_rev.items()}
