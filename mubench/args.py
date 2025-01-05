from dataclasses import dataclass, field
from typing import *


@dataclass
class UnlearningArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    unlearn_method: str = field(
        default='neggrad', metadata={"help": 'unlearning method'}
    )
    del_ratio: float = field(
        default=1.0, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    alpha: float = field(
        default=0.5, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    data_name: float = field(
        default=None, 
    )
    backbone: str = field(
        default=None, 
    )
    random_seed: int = field(
        default=None, 
    )
    use_cl: bool = field(
        default=False, metadata={"help": "Whether to use curriculum learning."}
    )
    use_lora: bool = field(
        default=False, metadata={"help": "Whether to use LoRA."}
    )
    lora_ratio: int = field(
        default=10, metadata={"help": "Ratio of LoRA parameters in percentage."}
    )
    use_mode_connectivity: bool = field(
        default=False, metadata={"help": "Whether to find mode connectivity curves with two unlearned models."}
    )
    mc_curve_type: str = field(
        default='linear', metadata={"help": "Type of mode connectivity curves."}
    )
    mc_num_bends: int = field(
        default=3, metadata={"help": "Number of bends for mode connectivity curve."}
    )

    def __post_init__(self):
        self.unlearn_method = self.unlearn_method.lower()
        # self.del_ratio = self.del_ratio / 100

