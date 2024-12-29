import os
import copy
import time
import math
import wandb
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import DataLoader
from typing import *
from transformers import Trainer, Seq2SeqTrainer
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import speed_metrics

from mubench.evaluation import Evaluator, TextGenEvaluator
from ..superloss import SuperLoss
from ..utils import load_base_model

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]


super_loss = SuperLoss('sl', lam=10, mode='avg')

def calculate_superloss(b_loss, batch):
    conf, tau, tau_adjusted = super_loss(b_loss, None, None)
    tau = [tau] * b_loss.shape[0]
    tau_adjusted = [tau_adjusted] * b_loss.shape[0]
    sl_loss = b_loss * conf

    return sl_loss


class UnlearningTrainer(Trainer):
    def __init__(self, **kwargs):
        self.raw_datasets = kwargs['raw_datasets']  # Used for computing performance on Df and Dr
        kwargs.pop('raw_datasets')
        self.unlearn_config = kwargs['unlearn_config']
        kwargs.pop('unlearn_config')
        self.unlearn_evaluator = Evaluator(self.unlearn_config)

        # Load original model
        kwargs['tokenizer'], kwargs['model'] = load_base_model(self.unlearn_config)

        super().__init__(**kwargs)
        self.num_labels = self.model.config.num_labels if hasattr(self.model.config, 'num_labels') else None
        self.unlearn_time = None
        self.method_specific_setup()

    def method_specific_setup(self):
        raise NotImplementedError

    def get_data(self, split_name):
        if split_name == 'eval':
            if 'validation' in self.raw_datasets:
                return self.raw_datasets['validation']
            else:
                return self.raw_datasets['test']

        return self.raw_datasets[split_name]

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss wrapper for unlearning method FineTune
        """
        if self.unlearn_config.use_cl:
            return self.compute_loss_cl(model, inputs, return_outputs)
        else:
            return self.compute_loss_non_cl(model, inputs, return_outputs)

    def compute_loss_non_cl(self, model, inputs, return_outputs=False):
        """
        Compute loss wrapper for unlearning method FineTune
        """
        # inputs = {k[len('dr_'):]: v for k, v in inputs.items() if k.startswith('dr_')}
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=return_outputs)
        else:
            loss = super().compute_loss(model, inputs, return_outputs=return_outputs)

        return (loss, outputs) if return_outputs else loss

    def compute_loss_cl(self, model, inputs, return_outputs=False):
        """
        Compute loss wrapper for curriculum learning (per-sample loss)
        """
        # inputs = {k[len('dr_'):]: v for k, v in inputs.items() if k.startswith('dr_')}
        if return_outputs:
            per_sample_loss, outputs = super().compute_loss(model, inputs, return_outputs=return_outputs)
        else:
            per_sample_loss = super().compute_loss(model, inputs, return_outputs=return_outputs)

        loss = calculate_superloss(per_sample_loss, inputs).mean()

        return (loss, outputs) if return_outputs else loss

    def unlearn(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs):

        self.unlearn_start_time = time.time()

        out = self.method_specific_unlearn(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

        self.unlearn_end_time = time.time()
        self.unlearn_time = self.unlearn_end_time - self.unlearn_start_time

        return out

    def method_specific_unlearn(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs):

        out = super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

        return out

    def get_df_logit(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="df"):
        start_time = time.time()
        
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation on Df",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        out_path = os.path.join(self.args.output_dir, f'pred_logit_df')
        np.save(out_path, output.predictions)

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        # self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", split_name=None):
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # if split_name is not None:
        out_path = os.path.join(self.args.output_dir, f'pred_logit_{metric_key_prefix}')
        np.save(out_path, output.predictions)

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        ## Update unlearning metrics
        # test_logit = np.load(f'checkpoint/{model_map_rev[model_args.model_name_or_path]}/{data_args.dataset_name}_{data_args.seed}/pred_logit_validation.npy')

        if metric_key_prefix is not None:

            # This is the current evaluation split. Can be validation / test set
            dt_logit = output.predictions
            dt_label = output.label_ids

            # This is the performance on Df
            df_output = self.get_df_logit(self.raw_datasets['df'])
            df_logit = df_output.predictions
            df_label = df_output.label_ids

            unlearn_metrics = self.unlearn_evaluator.compute(dt_logit, dt_label, df_logit, df_label)
            unlearn_metrics[metric_key_prefix + '_' + self.args.metric_for_best_model] = unlearn_metrics[self.args.metric_for_best_model]
            unlearn_metrics['unlearn_time'] = self.unlearn_time
            output.metrics.update(unlearn_metrics)


        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def evaluate_unlearn(self, ignore_keys=None):
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        start_time = time.time()

        # Performance on Dt, Df, Dr, OOD
        all_logit_and_label = []
        for split in ['test', 'df', 'dr', 'ood']:
            if split not in self.raw_datasets:
                continue
            eval_dataloader = self.get_eval_dataloader(self.raw_datasets[split])
            output = self.evaluation_loop(
                eval_dataloader, 
                description=split.capitalize(), 
                ignore_keys=ignore_keys, 
                metric_key_prefix=split
            )
            np.save(os.path.join(self.args.output_dir, f'pred_logit_{split}'), output.predictions)
            all_logit_and_label.extend([output.predictions, output.label_ids])
            self.log_metrics(split, output.metrics)
            self.save_metrics(split, output.metrics)

        unlearn_metrics = self.unlearn_evaluator.compute(*all_logit_and_label)
        if self.unlearn_time is not None:
            unlearn_metrics['unlearn_time'] = self.unlearn_time
        else:       # Read cached unlearn_time if this is only an evaluation
            if os.path.exists(os.path.join(self.args.output_dir, 'unlearn_final_results.json')):
                with open(os.path.exists(os.path.join(self.args.output_dir, 'unlearn_final_results.json'))) as f:
                    log = json.load(f)
                    if 'unlearn_time' in log and log['unlearn_time'] != -1:
                        unlearn_metrics['unlearn_time'] = -1
                    else:
                        unlearn_metrics['unlearn_time'] = log['unlearn_time']
            else:
                unlearn_metrics['unlearn_time'] = -1
        self.log_metrics('unlearn_final', unlearn_metrics)
        self.save_metrics('unlearn_final', unlearn_metrics)

        return unlearn_metrics

class UnlearningSeq2SeqTrainer(UnlearningTrainer, Seq2SeqTrainer):

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **gen_kwargs):

        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        # We don't want to drop samples in general
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs

        
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        out_path = os.path.join(self.args.output_dir, f'pred_logit_{metric_key_prefix}')
        np.save(out_path, output.predictions)

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        ## Update unlearning metrics
        # test_logit = np.load(f'checkpoint/{model_map_rev[model_args.model_name_or_path]}/{data_args.dataset_name}_{data_args.seed}/pred_logit_validation.npy')

        if metric_key_prefix is not None and metric_key_prefix != 'train':
            test_logit = output.predictions
            # test_label = self.get_data(metric_key_prefix)['label']
            test_label = output.label_ids
            df_output = self.get_df_logit(self.get_data('df'))
            df_logit = df_output.predictions
            # df_label = self.get_data('df')['label']
            df_label = df_output.label_ids
            # df_label = self.get_data('df')['label' if self.unlearn_config.unlearn_method != 'random_label' else 'ori_label']

            evaluator = TextGenEvaluator(None, test_label, test_logit, df_label, df_logit, dr_label=None, dr_pred=None, df_mask=None, tokenizer=self.tokenizer, metric_names=['rouge'])
            unlearn_metrics = evaluator.compute()
            unlearn_metrics[metric_key_prefix + '_' + self.args.metric_for_best_model] = unlearn_metrics[self.args.metric_for_best_model]
            unlearn_metrics['unlearn_time'] = self.unlearn_time if self.unlearn_time is not None else -1
            output.metrics.update(unlearn_metrics)


        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
