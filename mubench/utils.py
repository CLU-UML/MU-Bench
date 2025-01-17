import os
import mubench
from .curves import *


def load_base_model(unlearn_config):
    """
    Load the original model / base model for unlearning.

    Args:
        unlearn_config: config of unlearning

    Returns:
        tokenizer: The tokenizer for the model (if applicable).
        model: The loaded Hugging Face model.
    """

    model_cls_map = mubench.model_cls_map if not unlearn_config.use_cl else mubench.model_cls_cl_map

    if unlearn_config.data_name == 'nlvr2':
        # Check if the dataset is in the map
        if unlearn_config.backbone not in model_cls_map:
            raise ValueError(f"Backbone name '{unlearn_config.backbone}' not recognized.")

        tokenizer_class = AutoProcessor
        model_class = model_cls_map[unlearn_config.backbone]

        final_ckpt_path = 'dandelin/vilt-b32-finetuned-nlvr2'

    elif unlearn_config.data_name == 'tofu':
        from peft import LoraConfig, get_peft_model, PeftModel

        model = model_cls_map['tofu'].from_pretrained(
            f'locuslab/tofu_ft_{unlearn_config.backbone}',
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        config = LoraConfig(
            r=8, 
            lora_alpha=32, 
            target_modules=["q_proj","v_proj"], 
            lora_dropout=0.05,
            bias="none", 
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

        return model

    else:
        # Check if the dataset is in the map
        if unlearn_config.data_name not in model_cls_map:
            raise ValueError(f"Dataset name '{unlearn_config.data_name}' not recognized.")

        model_class = model_cls_map[unlearn_config.data_name]

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

    model = model_class.from_pretrained(final_ckpt_path)

    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        model.generation_config.do_sample = True

    return model

def init_curve(curve_type):
    pass

def load_base_model_mode_connectivity(unlearn_config):
    """
    Load the original model / base model for unlearning.

    Args:
        unlearn_config: config of unlearning

    Returns:
        tokenizer: The tokenizer for the model (if applicable).
        model: The loaded Hugging Face model.
    """

    num_labels_map = {'cifar100': 10, 'cifar100': 100, 'imdb': 2, 'ddi': 5, 'nlvr2': 3}
    dataset_to_model_map = {
        "cifar10": (AutoImageProcessor, AutoModelForImageClassification),
        "cifar100": (AutoImageProcessor, AutoModelForImageClassification),
        "imdb": (AutoTokenizer, AutoModelForSequenceClassification),
        "ddi": (AutoTokenizer, AutoModelForSequenceClassification),
        "ddi2013": (AutoTokenizer, AutoModelForSequenceClassification),
        "speech_commands": (AutoTokenizer, AutoModelForAudioClassification),
        "ucf101": (AutoTokenizer, AutoModelForVideoClassification),
        "samsum": (AutoTokenizer, AutoModelForSeq2SeqLM),
        "celeb_profile": (AutoTokenizer, AutoModelForCausalLM),
        "tiny_imagenet": (AutoImageProcessor, AutoModelForImageClassification),
        "tofu": (AutoTokenizer, AutoModelForCausalLM),
    }

    if unlearn_config.data_name == 'nlvr2':
        dataset_to_model_map = {
            "vilt": ViltForImagesAndTextClassification,
        }

        # Check if the dataset is in the map
        if unlearn_config.backbone not in dataset_to_model_map:
            raise ValueError(f"Backbone name '{unlearn_config.backbone}' not recognized.")

        tokenizer_class = AutoProcessor
        model_class = dataset_to_model_map[unlearn_config.backbone]

        final_ckpt_path = 'dandelin/vilt-b32-finetuned-nlvr2'

    else:
        # Check if the dataset is in the map
        if unlearn_config.data_name not in dataset_to_model_map:
            raise ValueError(f"Dataset name '{unlearn_config.data_name}' not recognized.")

        tokenizer_class, model_class = dataset_to_model_map[unlearn_config.data_name]

        # Check if a local checkpoint exists
        base_model_path = f'../../checkpoint/unlearn/{unlearn_config.data_name}/{unlearn_config.backbone}/{unlearn_config.unlearn_method}/{unlearn_config.del_ratio}'

    # Endpoints
    init_start = os.path.join(base_model_path, '42')
    init_end = os.path.join(base_model_path, '87')

    # This is the curve
    model = CurveModel(model_class, unlearn_config.mc_curve_type, init_start, init_end, unlearn_config.mc_num_bends)
    tokenizer = tokenizer_class.from_pretrained(init_start)

    if unlearn_config.data_name == 'nlvr2':
        return None, model

    return tokenizer, model
