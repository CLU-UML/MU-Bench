import os
from transformers import (
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoModelForAudioClassification,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    AutoModelForVideoClassification,
    AutoTokenizer,
)

def load_base_model(unlearn_config):
    """
    Load the original model / base model for unlearning.

    Args:
        unlearn_config: config of unlearning

    Returns:
        tokenizer: The tokenizer for the model (if applicable).
        model: The loaded Hugging Face model.
    """

    dataset_to_model_map = {
        "cifar100": AutoModelForImageClassification,
        "imdb": AutoModelForSequenceClassification,
        "ddi2013": AutoModelForTokenClassification,
        "nlvr2": AutoModelForVision2Seq,
        "speech_commands": AutoModelForAudioClassification,
        "ucf101": AutoModelForVideoClassification,
        "samsum": AutoModelForSeq2SeqLM,
        "celeb_profile": AutoModelForCausalLM,
        "tiny_imagenet": AutoModelForImageClassification,
    }

    # Check if the dataset is in the map
    if unlearn_config.data_name not in dataset_to_model_map:
        raise ValueError(f"Dataset name '{dataset_name}' not recognized.")

    tokenizer_class, model_class = dataset_to_model_map[unlearn_config.data_name]

    # Check if a local checkpoint exists
    base_model_path = f'{unlearn_config.data_name}/{unlearn_config.backbone}/{unlearn_config.data_name}_42'
    potential_local_ckpt = [
        os.path.join('checkpoint', base_model_path),
        os.path.join('../checkpoint', base_model_path),
        os.path.join('../../checkpoint', base_model_path),
    ]
    local_ckpt = False
    for path in potential_local_ckpt:
        if os.path.exists(path):
            tokenizer = tokenizer_class.from_pretrained(path)
            model = model_class.from_pretrained(path)
            local_ckpt = True
            break

    # Download from the MU-Bench repo on HuggingFace
    if not local_ckpt:
        tokenizer = tokenizer_class.from_pretrained(base_model_path)
        model = model_class.from_pretrained(base_model_path)

    return tokenizer, model
