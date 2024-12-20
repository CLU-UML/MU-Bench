# MU-Bench: A Multitask Multimodal Benchmark for Machine Unlearning

<!-- <p align="center">
    <a href="https://github.com/beir-cellar/beir/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/beir-cellar/beir.svg">
    </a>
    <a href="https://github.com/beir-cellar/beir/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/beir-cellar/beir.svg?color=green">
    </a>
    <a href="https://pepy.tech/project/beir">
        <img alt="Downloads" src="https://static.pepy.tech/personalized-badge/beir?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads">
    </a>
</p> -->

<h4 align="center">
    <p>
        <a href="https://arxiv.org/abs/2406.14796">📄 Paper</a> |
        <a href="https://clu-uml.github.io/MU-Bench-Project-Page">🌐 Project Page</a> |
        <a href="#Installation">⚙️ Installation</a> |
        <a href="#Usage">🚀 Example Usage</a> |
        <a href="https://colab.research.google.com/drive/1WiCYcYQ9T2eEPep38szZwPKgYEx-TGHD?usp=sharing">⚡ Demo</a>
    <p>
        <a href="https://huggingface.co/spaces/jialicheng/MU-Bench_Leaderboard">🏆 Leaderboard</a> |
        <a href="#Datasets">📊 Datasets</a> |
        <a href="https://docs.google.com/forms/d/e/1FAIpQLSfvCNaMy8H0-akM7DT4VoVOxLN_Qtd-wFre-EEYAPiCKC82xA/viewform?usp=header">📊 Add A New Dataset</a> |
        <a href="https://huggingface.co/collections/jialicheng/mu-bench-benchmarking-machine-unlearning-664a3dd153317bdff3d2fe45">🤗 Base Models</a>
    <p>
</h4>



## Introduction

MU-Bench is trying to address the following limitations of the evaluation of Machine Unlearning (MU):
1. Inconsistency. employing different trained models and architectures, and sample removal strategies, which hampers accurate comparison. In addition, prior MU approaches have mainly focused on _singular_ tasks or modalities, which is not comprehensive. To address these limitations, 

MU-Bench is the first comprehensive benchmark for MU that

1. unifies the sets of deleted samples and trained models
2. provides broad coverage of tasks and data modalities, including previously unexplored domains such as speech and video classification. 


## Datasets

To add a new dataset to `mubench`, please fill out this [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfvCNaMy8H0-akM7DT4VoVOxLN_Qtd-wFre-EEYAPiCKC82xA/viewform?usp=header) or concat the authors.


| **Dataset**              | **Task**                 | **Domain**          | **Modality**          | **D**    |
|--------------------------|--------------------------|---------------------|-----------------------|----------|
| **Discriminative Tasks** |                          |                     |                       |          |
| CIFAR-100                | Image Classification     | General             | Image                 | 50K      |
| IMDB                     | Sentiment Classification | Movie Review        | Text                  | 25K      |
| **DDI-2013**             | Relation Extraction      | Biomedical          | Text                  | 25K      |
| NLVR²                    | Visual Reasoning         | General             | Image-Image-Text      | 62K      |
| **Speech Commands**      | Keyword Spotting         | Commands            | Speech                | 85K      |
| **UCF101**               | Action Classification    | General             | Video                 | 9.3K     |
| **Generative Tasks**     |                          |                     |                       |          |
| SAMSum                   | Text Summarization       | Chat Dialogue       | Text                  | 14K      |
| **Celeb Profile**        | Text Generation          | Biography           | Text                  | 183      |
| **Tiny ImageNet**        | Text-to-Image Generation | General             | Image-Text            | 20K      |

**Bold** datasets are ones that have never been evaluated in unlearning.



## Installation

```Bash
pip install mubench
```

## Usage

#### Case 1: Access standardized data and base models from `mubench`

```python
import mubench
from mubench import UnlearningArguments, get_base_model, load_unlearn_data

unlearn_config = UnlearningArguments(
    unlearn_method="multi_delete",  # MU method, MultiDelete ECCV'24
    backbone="vilt",                # Network architecture
    data_name="nlvr2",              # Dataset
    del_ratio=5                     # Standardized splits
)

model, tokenizer = get_base_model(unlearn_config)
raw_datasets = load_unlearn_data(unlearn_config)

print(raw_datasets.keys())
```
By default, `load_unlearn_data` creates the training set `train` based on the unleaning method, as well as `df_eval` and `dr_eval` for evaluation. The original training set is `orig_train`.
```python
['train', 'validation', 'test', 'df_eval', 'dr_eval', 'orig_train']
```


#### Case 2: Unlearning with the standard data and base models from `mubench`
```python
# Standard HuggingFace code
from transformers import TrainingArguments
args = TrainingArguments(output_dir="tmp")

# Additional code for unlearning
from mubench import UnlearningArguments, unlearn_trainer
unlearn_config = UnlearningArguments(
    unlearn_method="multi_delete",  # MU method, MultiDelete ECCV'24
    backbone="vilt",                # Network architecture
    data_name="nlvr2",              # Dataset
    del_ratio=5                     # Standardized splits
)
trainer = unlearn_trainer(unlearn_config.unlearn_method)(
    args=args, 
    unlearn_config=unlearn_config
)
trainer.unlearn()                   # Start Unlearning and Evaluation!
```

#### Case 3: Unlearning with customized original models from `mubench`
```python
# Standard HuggingFace code
from transformers import TrainingArguments
args = TrainingArguments(output_dir="tmp")

# Additional code for unlearning
from mubench import UnlearningArguments, unlearn_trainer

model = # Define your own model
raw_datasets = load_unlearn_data(unlearn_config)
raw_datasets['train'] = # Customize unlearning data

unlearn_config = UnlearningArguments(
    unlearn_method="multi_delete",  # MU method, MultiDelete ECCV'24
    backbone="vilt",                # Network architecture
    data_name="nlvr2",              # Dataset
    del_ratio=5                     # Standardized splits
)
trainer = unlearn_trainer(unlearn_config.unlearn_method)(
    args=args, 
    unlearn_config=unlearn_config,
    model=model,                    # Overwrite the standard model
    raw_datasets=raw_datasets,      # Overwrite the standard data
)
trainer.unlearn()                   # Start Unlearning and Evaluation!
```

#### Case 3: Access data and models
```python
from mubench import get_training_, unlearn_trainer
unlearn_config = UnlearningArguments(
    unlearn_method="multi_delete",  # MU method, MultiDelete ECCV'24
    backbone="vilt",                # Network architecture
    data_name="nlvr2",              # Dataset
    del_ratio=5                     # Standardized splits
)
trainer = unlearn_trainer(unlearn_config.unlearn_method)(
    args=args, 
    unlearn_config=unlearn_config
)
trainer.unlearn()                   # Start Unlearning and Evaluation!
```

## 📢 Updates and Changelog

Stay informed about the latest developments and enhancements to this project. Below is a summary of recent updates:

### 📅 **Planned Updates**
- [Feature]: Implement more unlearning algorithms. Estimated release in [month/year].
- [Improvement]: Include more base models / architectures. Estimated release in [month/year].

<!-- **Note**: For a complete history, refer to the [Changelog](link-to-detailed-changelog). -->
