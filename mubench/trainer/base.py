import os
import copy
import time
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.utils.data import DataLoader
from typing import *
from transformers import Trainer, Seq2SeqTrainer
from transformers.trainer_utils import speed_metrics
from transformers.utils import logging

import mubench
from ..evaluation import Evaluator, TextGenEvaluator, compute_metrics_map
from ..superloss import SuperLoss
from ..data.preprocess import all_dataset_collators
from ..utils import load_base_model, load_base_model_mode_connectivity

logger = logging.get_logger(__name__)


class UnlearningTrainer(Trainer):
    def __init__(self, **kwargs):
        # Datasets
        self.raw_datasets = kwargs['raw_datasets']  # Used for computing performance on Df and Dr
        kwargs.pop('raw_datasets')
        kwargs['train_dataset'] = kwargs['train_dataset'] if 'train_dataset' in kwargs else self.raw_datasets['train']
        if 'eval_dataset' not in kwargs:
            if 'validation' in self.raw_datasets:
                kwargs['eval_dataset'] = self.raw_datasets['validation']
            elif 'test' in self.raw_datasets:
                kwargs['eval_dataset'] = self.raw_datasets['test']
                logger.warning('Using test set as eval_dataset.')
            else:
                logger.warning('No eval_dataset provided.')

        self.unlearn_config = kwargs['unlearn_config']
        kwargs.pop('unlearn_config')
        self.unlearn_evaluator = Evaluator(self.unlearn_config)

        # Initialize SuperLoss calculator for Curriculum Learning
        self.super_loss = SuperLoss('sl', lam=10, mode='avg') if self.unlearn_config.use_cl else None

        # Load original model
        if 'model' not in kwargs:
            kwargs['model'] = load_base_model(self.unlearn_config)

            if self.unlearn_config.use_mode_connectivity:
                kwargs['tokenizer'], kwargs['model'] = load_base_model_mode_connectivity(self.unlearn_config)

        if 'data_collator' not in kwargs:
            kwargs['data_collator'] = all_dataset_collators[self.unlearn_config.data_name] if self.unlearn_config.data_name in all_dataset_collators else None

        if 'compute_metrics' not in kwargs:
            kwargs['compute_metrics'] = compute_metrics_map[self.unlearn_config.data_name] if self.unlearn_config.data_name in compute_metrics_map else None

        super().__init__(**kwargs)
        self.num_labels = mubench.num_classes_map[self.unlearn_config.data_name] if self.unlearn_config.data_name in mubench.num_classes_map else None
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

    def calculate_superloss(self, b_loss, batch):
        conf, tau, tau_adjusted = self.super_loss(b_loss, None, None)
        tau = [tau] * b_loss.shape[0]
        tau_adjusted = [tau_adjusted] * b_loss.shape[0]
        sl_loss = b_loss * conf

        return sl_loss

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
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=return_outputs)
        else:
            loss = super().compute_loss(model, inputs, return_outputs=return_outputs)

        return (loss, outputs) if return_outputs else loss

    def compute_loss_cl(self, model, inputs, return_outputs=False):
        """
        Compute loss wrapper for curriculum learning (per-sample loss)
        """
        if return_outputs:
            per_sample_loss, outputs = super().compute_loss(model, inputs, return_outputs=return_outputs)
        else:
            per_sample_loss = super().compute_loss(model, inputs, return_outputs=return_outputs)

        loss = self.calculate_superloss(per_sample_loss, inputs).mean()

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
        if self.unlearn_config.use_mode_connectivity:
            # Only use Dt and Df for evaluation during training
            # Use all splits in the final testing
            metrics = self.evaluate_mode_connectivity_curve(['test', 'df'], 5)
            if metric_key_prefix is not None:
                metrics[metric_key_prefix + '_' + self.args.metric_for_best_model] = metrics[self.args.metric_for_best_model]
                metrics['unlearn_time'] = self.unlearn_time

            return metrics

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

    def evaluate_unlearn(self, eval_splits=['test', 'df', 'dr', 'ood'], ignore_keys=None):
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        start_time = time.time()

        # Performance on Dt, Df, Dr, OOD
        all_logit_and_label = []
        for split in eval_splits:
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
        # self.log_metrics('unlearn_final', unlearn_metrics)
        # self.save_metrics('unlearn_final', unlearn_metrics)

        return unlearn_metrics

    def evaluate_mode_connectivity_curve(self, eval_splits=['test', 'df', 'dr', 'ood'], num_points=61):
        all_metric = []
        ts = np.linspace(0, 1, num_points)
        for t in tqdm(ts, desc='Points on Curve'):
            t = torch.tensor(t)
            interpolated_params = self.model.interpolate_weights(t)
            self.model.final_model.load_state_dict(interpolated_params, strict=False)
            print(f'Eval with interpolated weights, t = {t} and weight = {self.model.curve(t).detach().tolist()}')

            metrics = self.evaluate_unlearn(eval_splits)
            all_metric.append(copy.deepcopy(metrics))

        all_metric = pd.DataFrame(all_metric)
        all_metric['t'] = ts
        all_metric.to_csv(os.path.join(self.args.output_dir, 'eval_curve.csv'), index=None)

        return all_metric.mean(0).to_dict()


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
