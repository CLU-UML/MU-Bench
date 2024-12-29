orig_acc = {
    'cifar100': {
        'random': 1/10,
        'resnet-50': {'dt': 0.8328, 'dr': 0.9156, 'ood': 0},
        'swin-tiny': {'dt': 0.8714, 'dr': 0.9957411764705882, 'ood': 0},
        'swin-base': {'dt': 0.9301, 'dr': 9971764705882353, 'ood': 0},
        'vit-base': {'dt': 0.922, 'dr': 0.9849176470588236, 'ood': 0},
        'vit-large': {'dt': 0.9356, 'dr': 9972470588235294, 'ood': 0},
    },
    
    'imdb': {
        'random': 1/2,
        # 'bert-base': {'dt': 'dr': 'ood':},
        # 'bert-large': {'dt': 'dr': 'ood':},
        # 'roberta-base': {'dt': 'dr': 'ood':},
        # 'roberta-large': {'dt': 'dr': 'ood':},
        # 'bert-base': {'dt': 'dr': 'ood':},
    }
    
    # 'nlvr2': {'dt': 'dr': 'random': 1/3},
}