import os
import copy
import time
import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from typing import *
from transformers import Trainer, Seq2SeqTrainer

from .base import calculate_superloss, UnlearningTrainer

from transformers.utils import logging

logger = logging.get_logger(__name__)


def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children


class BadTeachingTrainer(UnlearningTrainer):
        
    def method_specific_setup(self):
        self.good_teacher = copy.deepcopy(self.model)
        self.bad_teacher = copy.deepcopy(self.model)

        layers = get_children(self.bad_teacher)
        _ = [l.reset_parameters() for l in layers if hasattr(l, 'reset_parameters')]

        for (n1, p1), (n2, p2) in zip(self.good_teacher.named_parameters(), self.bad_teacher.named_parameters()):
            assert n1 == n2
            if n1 == 'wav2vec2.masked_spec_embed':   # For wav2vec2
                continue
            if not p1.requires_grad:
                continue
            print(f'bad teacher {n1} same as original?', (p1 == p2).all())
            # assert not (p1 == p2).all(), f"{n1}, {n2}"
            p1.requires_grad = False
            p2.requires_grad = False

        self.good_teacher.eval()
        self.bad_teacher.eval()

    def compute_loss_cl(self, model, inputs, return_outputs=False):
        """
        Compute loss wrapper for unlearning method FineTune
        """
        if model.training:
            # Unlearned model
            input_df_mask = inputs['is_df'] == 1
            inputs.pop('is_df')
            dr_inputs = {k: v[~input_df_mask] for k, v in inputs.items()}
            # dr_inputs = {k[len('dr_'):]: v for k, v in inputs.items() if k.startswith('dr_')}
            if 'labels' in dr_inputs:
                dr_inputs.pop('labels')
            if 'label_ids' in dr_inputs:
                dr_inputs.pop('label_ids')
            
            dr_outputs = model(**dr_inputs, return_dict=True)
            with torch.no_grad():
                good_outputs = self.good_teacher(**dr_inputs, return_dict=True)

            df_inputs = {k: v[input_df_mask] for k, v in inputs.items()}
            # df_inputs = {k[len('df_'):]: v for k, v in inputs.items() if k.startswith('df_')}
            if 'labels' in df_inputs:
                df_inputs.pop('labels')
            if 'label_ids' in df_inputs:
                df_inputs.pop('label_ids')

            df_outputs = model(**df_inputs, return_dict=True)
            with torch.no_grad():
                bad_outputs = self.bad_teacher(**df_inputs, return_dict=True)

            kl_loss = nn.KLDivLoss(reduction='none' if self.use_cl else "batchmean")

            dr_loss = kl_loss(dr_outputs.logits, good_outputs.logits)
            df_loss = kl_loss(df_outputs.logits, bad_outputs.logits)

            dr_loss = calculate_superloss(dr_loss, dr_inputs).mean()
            df_loss = calculate_superloss(df_loss, df_inputs).mean()

            loss = dr_loss + df_loss
            outputs = df_outputs

        else:
            if 'is_df' in inputs:
                inputs.pop('is_df')
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            loss = loss.mean()

        return (loss, outputs) if return_outputs else loss

    def compute_loss_non_cl(self, model, inputs, return_outputs=False):
        """wrapper for compute_loss training with non-SuperLoss."""

        if model.training:
            # Unlearned model
            input_df_mask = inputs['is_df'] == 1
            inputs.pop('is_df')
            dr_inputs = {k: v[~input_df_mask] for k, v in inputs.items()}
            # dr_inputs = {k[len('dr_'):]: v for k, v in inputs.items() if k.startswith('dr_')}
            if 'labels' in dr_inputs:
                dr_inputs.pop('labels')
            if 'label_ids' in dr_inputs:
                dr_inputs.pop('label_ids')
            
            dr_outputs = model(**dr_inputs, return_dict=True)
            with torch.no_grad():
                good_outputs = self.good_teacher(**dr_inputs, return_dict=True)

            df_inputs = {k: v[input_df_mask] for k, v in inputs.items()}
            # df_inputs = {k[len('df_'):]: v for k, v in inputs.items() if k.startswith('df_')}
            if 'labels' in df_inputs:
                df_inputs.pop('labels')
            if 'label_ids' in df_inputs:
                df_inputs.pop('label_ids')

            df_outputs = model(**df_inputs, return_dict=True)
            with torch.no_grad():
                bad_outputs = self.bad_teacher(**df_inputs, return_dict=True)

            kl_loss = nn.KLDivLoss(reduction="batchmean")
            dr_loss = kl_loss(dr_outputs.logits, good_outputs.logits)
            df_loss = kl_loss(df_outputs.logits, bad_outputs.logits)
            loss = dr_loss + df_loss
            outputs = df_outputs

        else:
            if 'is_df' in inputs:
                inputs.pop('is_df')
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


class _BadTeachingTrainer(UnlearningTrainer):
    def prepare_dr_dataloader(self):
        train_dataset = self.raw_datasets['dr']
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return iter(DataLoader(train_dataset, **dataloader_params))

    def unlearn(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs):

        self.unlearn_start_time = time.time()

        self.dr_loader = self.prepare_dr_dataloader()
        out = super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

        self.unlearn_end_time = time.time()
        self.unlearn_time = self.unlearn_end_time - self.unlearn_start_time

        return out

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.use_cl:
            return self.compute_loss_cl(model, inputs, return_outputs)
        else:
            return self.compute_loss_non_cl(model, inputs, return_outputs)

    def compute_loss_non_cl(self, model, inputs, return_outputs=False):
        if model.training:
            # Unlearned model
            df_inputs = inputs
            dr_inputs = next(self.dr_loader)
            dr_inputs = self._prepare_inputs(dr_inputs)

            if 'labels' in dr_inputs:
                dr_inputs.pop('labels')
            if 'label_ids' in dr_inputs:
                dr_inputs.pop('label_ids')
            
            dr_outputs = model(**dr_inputs, return_dict=True)
            with torch.no_grad():
                good_outputs = self.good_teacher(**dr_inputs, return_dict=True)

            if 'labels' in df_inputs:
                df_inputs.pop('labels')
            if 'label_ids' in df_inputs:
                df_inputs.pop('label_ids')

            df_outputs = model(**df_inputs, return_dict=True)
            with torch.no_grad():
                bad_outputs = self.bad_teacher(**df_inputs, return_dict=True)

            kl_loss = nn.KLDivLoss(reduction="batchmean")
            dr_loss = kl_loss(dr_outputs.logits, good_outputs.logits)
            df_loss = kl_loss(df_outputs.logits, bad_outputs.logits)
            loss = dr_loss + df_loss
            outputs = df_outputs

        else:
            if 'is_df' in inputs:
                inputs.pop('is_df')
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def compute_loss_cl(self, model, inputs, return_outputs=False):
        if model.training:
            # Unlearned model
            df_inputs = inputs
            dr_inputs = next(self.dr_loader)
            dr_inputs = self._prepare_inputs(dr_inputs)

            if 'labels' in dr_inputs:
                dr_inputs.pop('labels')
            if 'label_ids' in dr_inputs:
                dr_inputs.pop('label_ids')
            
            dr_outputs = model(**dr_inputs, return_dict=True)
            with torch.no_grad():
                good_outputs = self.good_teacher(**dr_inputs, return_dict=True)

            if 'labels' in df_inputs:
                df_inputs.pop('labels')
            if 'label_ids' in df_inputs:
                df_inputs.pop('label_ids')

            df_outputs = model(**df_inputs, return_dict=True)
            with torch.no_grad():
                bad_outputs = self.bad_teacher(**df_inputs, return_dict=True)

            kl_loss = nn.KLDivLoss(reduction="none")
            dr_loss = kl_loss(dr_outputs.logits, good_outputs.logits).mean(axis=1)
            df_loss = kl_loss(df_outputs.logits, bad_outputs.logits).mean(axis=1)
            dr_loss = calculate_superloss(dr_loss, dr_inputs).mean()
            df_loss = calculate_superloss(df_loss, df_inputs).mean()
            loss = dr_loss + df_loss
            outputs = df_outputs

        else:
            if 'is_df' in inputs:
                inputs.pop('is_df')
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            loss = loss.mean()

        return (loss, outputs) if return_outputs else loss

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

        if self.unlearning and metric_key_prefix is not None and metric_key_prefix != 'train':
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
