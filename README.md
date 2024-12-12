# MU-Bench: Benchmarking Machine Unlearning

<p align="center">
    <a href="https://github.com/beir-cellar/beir/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/beir-cellar/beir.svg">
    </a>
    <a href="https://github.com/beir-cellar/beir/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/beir-cellar/beir.svg?color=green">
    </a>
    <a href="https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing">
        <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
    <a href="https://pepy.tech/project/beir">
        <img alt="Downloads" src="https://static.pepy.tech/personalized-badge/beir?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="https://arxiv.org/abs/2406.14796">📄 Paper</a> |
        <a href="https://clu-uml.github.io/MU-Bench-Project-Page">🌐 Project Page</a> |
        <a href="">🏆 Leaderboard</a> |
        <a href="#beers-installation">⚙️ Installation</a> |
        <a href="#beers-quick-example">🚀 Quick Example</a> |
        <a href="#Datasets">📊 Datasets</a> |
        <a href="https://huggingface.co/BeIR">🤗 Hugging Face</a>
    <p>
</h4>



## Introduction


## Datasets

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

## How to use

```python
# Standard HuggingFace code
from transformers import TrainingArguments, AutoTokenizer, ViltForImagesAndTextClassification
args = TrainingArguments(output_dir="tmp")

# Additional code for unlearning
from mubench import UnlearningArguments, UnlearningTrainer
unlearn_config = UnlearningArguments(
    unlearn_method="multi_delete",  # MU method
    backbone="vilt",                # Network architecture
    data_name="nlvr2",              # Dataset
    del_ratio=5                     # Standardized splits
)
trainer = UnlearningTrainer(
    args=args, 
    unlearn_config=unlearn_config
)
trainer.unlearn()                   # Start Unlearning and Evaluation!
```


##

## 📢 Updates and Changelog

Stay informed about the latest developments and enhancements to this project. Below is a summary of recent updates:

### 📅 **Planned Updates**
- [Feature]: Implement more unlearning algorithms. Estimated release in [month/year].
- [Improvement]: Include more base models / architectures. Estimated release in [month/year].

<!-- **Note**: For a complete history, refer to the [Changelog](link-to-detailed-changelog). -->
