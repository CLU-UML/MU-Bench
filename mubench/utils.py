import os
from transformers import (
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoModelForAudioClassification,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    AutoModelForVideoClassification,
    AutoModelForSeq2SeqLM,
    AutoImageProcessor,
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
        "cifar100": (AutoImageProcessor, AutoModelForImageClassification),
        "imdb": (AutoTokenizer, AutoModelForSequenceClassification),
        "ddi2013": (AutoTokenizer, AutoModelForTokenClassification),
        "nlvr2": (AutoTokenizer, AutoModelForVision2Seq),
        "speech_commands": (AutoTokenizer, AutoModelForAudioClassification),
        "ucf101": (AutoTokenizer, AutoModelForVideoClassification),
        "samsum": (AutoTokenizer, AutoModelForSeq2SeqLM),
        "celeb_profile": (AutoTokenizer, AutoModelForCausalLM),
        "tiny_imagenet": (AutoImageProcessor, AutoModelForImageClassification),
    }

    # Check if the dataset is in the map
    if unlearn_config.data_name not in dataset_to_model_map:
        raise ValueError(f"Dataset name '{dataset_name}' not recognized.")

    tokenizer_class, model_class = dataset_to_model_map[unlearn_config.data_name]

    # Check if a local checkpoint exists
    base_model_path = f'{unlearn_config.data_name}/{unlearn_config.backbone}'
    potential_local_ckpt = [
        os.path.join('checkpoint', unlearn_config.backbone),
        os.path.join('../checkpoint', base_model_path),
        os.path.join('../../checkpoint', base_model_path),
    ]
    final_ckpt_path = None
    for path in potential_local_ckpt:
        if os.path.exists(path):
            final_ckpt_path = path
            break

    # Download from the MU-Bench repo on HuggingFace
    if final_ckpt_path is None:
        final_ckpt_path = f'jialicheng/{unlearn_config.data_name}-{unlearn_config.backbone}'

    tokenizer = tokenizer_class.from_pretrained(final_ckpt_path)
    model = model_class.from_pretrained(final_ckpt_path)

    return tokenizer, model
