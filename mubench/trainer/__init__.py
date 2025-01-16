from .bad_teaching import BadTeachingTrainer
from .grad_ascent import GradAscentTrainer
from .random_labeling import RandomLabelingTrainer
from .scrub import SCRUBTrainer
from .salun import SalUnTrainer
from .grad_diff import GradDiffTrainer
from .npo import NPOTrainer


def get_trainer(trainer_name):
    # Dictionary to map trainer names to their respective functions
    trainer_methods = {
        "bad_teaching": BadTeachingTrainer,
        "grad_ascent": GradAscentTrainer,
        "neggrad": GradAscentTrainer,
        "random_label": RandomLabelingTrainer,
        "scrub": SCRUBTrainer,
        "salun": SalUnTrainer,
        "grad_diff": GradDiffTrainer,
        "npo": NPOTrainer,
    }

    # Check if the trainer_name exists in the dictionary
    if trainer_name in trainer_methods:
        return trainer_methods[trainer_name]
    else:
        # Raise an error if the trainer_name is not recognized
        raise ValueError(f"Trainer '{trainer_name}' not found. Available trainers are: {', '.join(trainer_methods.keys())}")
