from peft import LoraConfig

kwargs = {'lora_dropout': 0.1, 'modules_to_save': ['classifier']}
UNET_TARGET_MODULES = ["to_q", "to_v", "query", "value"]

lora_config_mapping = {
    # Audio
    ('hubert-base', 10): LoraConfig(r=256, lora_alpha=256, target_modules=["q_proj", "v_proj"], **kwargs),
    ('hubert-base', 20): LoraConfig(r=768, lora_alpha=768, target_modules=["q_proj", "v_proj"], **kwargs),
    ('hubert-base', 50): LoraConfig(r=2560, lora_alpha=2560, target_modules=["q_proj", "v_proj"], **kwargs),

    # Text
    ('bert-base', 10): LoraConfig(r=384, lora_alpha=384, task_type="SEQ_CLS", **kwargs),
    ('bert-base', 20): LoraConfig(r=768, lora_alpha=768, task_type="SEQ_CLS", **kwargs),
    ('bert-base', 50): LoraConfig(r=3072, lora_alpha=3072, task_type="SEQ_CLS", **kwargs),

    ('biobert', 10): LoraConfig(r=384, lora_alpha=384, task_type="SEQ_CLS", **kwargs),
    ('biobert', 20): LoraConfig(r=768, lora_alpha=768, task_type="SEQ_CLS", **kwargs),
    ('biobert', 50): LoraConfig(r=3072, lora_alpha=3072, task_type="SEQ_CLS", **kwargs),

    # Image
    ('vit-base-patch16-224', 10): LoraConfig(r=256, lora_alpha=256, target_modules=["query", "value"], **kwargs),
    ('vit-base-patch16-224', 20): LoraConfig(r=512, lora_alpha=512, target_modules=["query", "value"], **kwargs),
    ('vit-base-patch16-224', 50): LoraConfig(r=2560, lora_alpha=2560, target_modules=["query", "value"], **kwargs),

    # Image-Text
    ('vilt', 10): LoraConfig(r=768, lora_alpha=768, target_modules=["query", "value"], **kwargs),
    ('vilt', 20): LoraConfig(r=768*2, lora_alpha=768*2, target_modules=["query", "value"], **kwargs),
    ('vilt', 50): LoraConfig(r=768*8, lora_alpha=768*8, target_modules=["query", "value"], **kwargs),

    # Video
    ('videomae-base', 10): LoraConfig(r=512, lora_alpha=512, target_modules=["query", "value"], **kwargs),
    ('videomae-base', 20): LoraConfig(r=1024, lora_alpha=1024, target_modules=["query", "value"], **kwargs),
    ('videomae-base', 50): LoraConfig(r=768*6, lora_alpha=768*6, target_modules=["query", "value"], **kwargs),

    # SD, https://github.com/huggingface/peft/blob/main/examples/lora_dreambooth/train_dreambooth.py
    ('sd', 10): LoraConfig(r=1024, lora_alpha=1024, target_modules=UNET_TARGET_MODULES, **kwargs),
    ('sd', 20): LoraConfig(r=2560, lora_alpha=2560, target_modules=UNET_TARGET_MODULES, **kwargs),
    ('sd', 50): LoraConfig(r=7680, lora_alpha=7680, target_modules=UNET_TARGET_MODULES, **kwargs),
}
