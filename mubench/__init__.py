from .args import UnlearningArguments
from .data.base import load_unlearn_data
from .utils import load_base_model
from .trainer import get_trainer


num_classes_map = {'cifar10': 10, 'cifar100': 100, 'imdb': 2, 'ddi': 5, 'nlvr2': 3}
metric_name_map = {'cifar10': 'accuracy', 'cifar100': 'accuracy', 'imdb': 'accuracy', 'ddi': 'accuracy', 'nlvr2': 'accuracy', 'samsum': 'rougeL'}

## Mapping of base model names (short name to full name in HuggingFace)
model_map_rev = {
    # CIFAR10, CIFAR100
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
    'bert-large-uncased': 'bert-large',
    'FacebookAI/roberta-base': 'roberta-base',
    'FacebookAI/roberta-large': 'roberta-large',
    'distilbert/distilbert-base-uncased': 'distilbert-base',
    'google/electra-base-discriminator': 'electra-base',
    'microsoft/deberta-v3-base': 'deberta-base',
    'albert/albert-base-v2': 'albert-base-v2',

    # DDI2013
    'dmis-lab/biobert-v1.1': 'biobert',
    'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract': 'pubmedbert-abstract',
    'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext': 'pubmedbert-fulltext',


    # NLVR2
    'dandelin/vilt-b32-finetuned-nlvr2': 'vilt',


    # Speech Commands
    'facebook/wav2vec2-base': 'wav2vec2-base',
    'facebook/wav2vec2-large': 'wav2vec2-large',
    'facebook/hubert-base-ls960': 'hubert-base',
    'facebook/hubert-large-ll60k': 'hubert-large',
    'facebook/hubert-xlarge-ll60k': 'hubert-xlarge',
    'openai/whisper-tiny': 'whisper-tiny',
    'openai/whisper-base': 'whisper-base',

    # TOFU
    'microsoft/phi-1_5': 'phi-1.5',
    'NousResearch/Llama-2-7b-chat-hf': 'llama2-7b',

}

model_map = {j: i for i, j in model_map_rev.items()}


# Mapping for model class
from transformers import (
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoModelForAudioClassification,
    ViltForImagesAndTextClassification,
    AutoModelForCausalLM,
    AutoModelForVideoClassification,
    AutoModelForSeq2SeqLM,
)
model_cls_map = {
    "cifar10": AutoModelForImageClassification,
    "cifar100": AutoModelForImageClassification,
    "imdb": AutoModelForSequenceClassification,
    "ddi": AutoModelForSequenceClassification,
    "ddi2013": AutoModelForSequenceClassification,
    "speech_commands": AutoModelForAudioClassification,
    "ucf101": AutoModelForVideoClassification,
    "samsum": AutoModelForSeq2SeqLM,
    "celeb_profile": AutoModelForCausalLM,
    "tiny_imagenet": AutoModelForImageClassification,
    "tofu": AutoModelForCausalLM,
    "vilt": ViltForImagesAndTextClassification,
}

# Mapping for model class using CL
from transformers_for_cl import (
    AutoModelForImageClassification as AutoModelForImageClassificationCL,
    AutoModelForSequenceClassification as AutoModelForSequenceClassificationCL,
    AutoModelForAudioClassification as AutoModelForAudioClassificationCL,
    ViltForImagesAndTextClassification as ViltForImagesAndTextClassificationCL,
    AutoModelForCausalLM as AutoModelForCausalLMCL,
    AutoModelForVideoClassification as AutoModelForVideoClassificationCL,
    AutoModelForSeq2SeqLM as AutoModelForSeq2SeqLMCL,
)

model_cls_cl_map = {
    "cifar10": AutoModelForImageClassificationCL,
    "cifar100": AutoModelForImageClassificationCL,
    "imdb": AutoModelForSequenceClassificationCL,
    "ddi": AutoModelForSequenceClassificationCL,
    "ddi2013": AutoModelForSequenceClassificationCL,
    "speech_commands": AutoModelForAudioClassificationCL,
    "ucf101": AutoModelForVideoClassificationCL,
    "samsum": AutoModelForSeq2SeqLMCL,
    "celeb_profile": AutoModelForCausalLMCL,
    "tiny_imagenet": AutoModelForImageClassificationCL,
    "tofu": AutoModelForCausalLMCL,
}
