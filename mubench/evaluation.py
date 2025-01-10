import os
import evaluate
import numpy as np
import nltk
from scipy.spatial.distance import jensenshannon 
from scipy.special import softmax 
from sklearn.metrics import roc_auc_score
from .original_performance import orig_acc


def harmonic_mean(dt_acc, df_acc, orig_dt_acc=1.0, random_acc=0.5):
    """
    Compute the harmonic mean of performance of dt_acc and abs(df_acc - random_acc).
    We aim to find an unlearned model with both close-to-original performance on Dt and close-to-random performance on Df.
    
    Args:
        dt_acc (float): performance on Dt. Higher better.
        df_acc (float): performance on Df. Lower better, but should not be lower than random performance 
                        (to be converted to abs difference from random accuracy).
        random_acc (float): random performance (default is 0.5 for binary classification).
    
    Returns:
        float: The harmonic mean of dt_acc and abs(df_acc - random_acc).
    """
    # Convert dt and df_acc
    diff_dt_acc = 1 - abs(dt_acc - orig_dt_acc)
    diff_df_acc = 1 - abs(df_acc - random_acc)
    
    harmonic_mean = 2 * diff_dt_acc * diff_df_acc / (diff_dt_acc + diff_df_acc)
    
    return harmonic_mean


class Evaluator:
    def __init__(self, unlearn_config, metric_names=['accuracy']):
        self.unlearn_config = unlearn_config
        self.metric_names = metric_names

        self.metrics = {}
        self.evaluator = {}
        for metric_name in self.metric_names:
            self.evaluator[metric_name] = evaluate.load(metric_name)

    def compute(self, dt_pred, dt_label, df_pred, df_label, dr_pred=None, dr_label=None, ood_pred=None, ood_label=None):
        for metric_name in self.metric_names:
            self.get_task_performance(dt_pred, dt_label, 'dt', metric_name)
            self.get_task_performance(df_pred, df_label, 'df', metric_name)
            
            if dr_pred is not None:
                self.get_task_performance(dr_pred, dr_label, 'dr', metric_name)
                self.get_knowledge_gap(dr_pred, dr_label, df_pred, df_label)
                self.get_zero_retrain_forgetting_score(df_pred)
                self.get_mia_score()

            if ood_pred is not None:
                self.get_task_performance(ood_pred, ood_label, 'ood', metric_name)

            dt_acc = self.metrics['dt_' + metric_name]
            df_acc = self.metrics['df_' + metric_name]
            orig_dt_acc = orig_acc[self.unlearn_config.data_name][self.unlearn_config.backbone]['dt']
            random_acc = orig_acc[self.unlearn_config.data_name]['random']

            # Use this metric for model selection
            overall_acc = harmonic_mean(dt_acc, df_acc, orig_dt_acc, random_acc)

            # We need some extra restrictions to prevent Dt / Df from dropping significantly
            if not self.unlearn_config.use_mode_connectivity:

                # dt_acc cannot drop by 20% compared to orig_dt_acc
                if abs(dt_acc - orig_dt_acc) / orig_dt_acc > 0.2:
                    overall_acc = 0

                # df_acc cannot drop below random_acc
                if df_acc < random_acc and (random_acc - df_acc) / random_acc > 0.2:
                    overall_acc = 0

            self.metrics['unlearn_overall_' + metric_name] = overall_acc

            # if dr_pred is not None:
            #     self.metrics['unlearn_overall_' + metric_name] = (
            #         self.metrics['unlearn_overall_' + metric_name] + self.metrics['knowledge_gap']
            #     ) / 3
            # else:
            #     self.metrics['unlearn_overall_' + metric_name] = self.metrics['unlearn_overall_' + metric_name] / 2

        return self.metrics

    def get_task_performance(self, logits, label, subset='test', metric_name='accuracy'):
        evaluator = self.evaluator[metric_name]
        metric_val = evaluator.compute(references=label, predictions=np.argmax(logits, axis=1))
        if metric_name == 'rouge':
            metric_name = 'rougeL'
        self.metrics[subset + '_' + metric_name] = metric_val[metric_name]

    def get_knowledge_gap(self, dr_pred, dr_label, df_pred, df_label):
        df_size = df_label.shape[0]
        binary_label = [1] * df_size + [0] * df_size

        gap = []
        all_idx = np.arange(dr_label.shape[0])
        for _ in range(500):
            sel_idx = np.random.choice(all_idx, df_size, replace=False)
            label = dr_label[sel_idx].tolist() + df_label.tolist()
            pred = np.hstack([np.argmax(dr_pred[sel_idx], axis=1), np.argmax(df_pred, axis=1)])
            binary_pred = np.array(label == pred).astype(int)
            auc = roc_auc_score(binary_label, binary_pred)
            gap.append(auc)

        print('D_f | D_r', np.mean(gap))
        self.metrics['knowledge_gap'] = np.mean(gap)

    def get_zero_retrain_forgetting_score(self, df_pred):
        zrfs = []
        for _ in range(500):
            # Generate probability for incompetent teacher
            logits = np.random.rand(*df_pred.shape)
            random_prob = softmax(logits, 1)

            df_prob = softmax(df_pred, 1)
            dis = jensenshannon(df_prob, random_prob, axis=0)
            div = dis ** 2
            zrf = 1 - div.mean()
            zrfs.append(zrf)
        
        print('ZRF', np.mean(zrfs))
        self.metrics['zrf'] = np.mean(zrfs)

    def get_mia_score(self,):
        pass


class TextGenEvaluator(Evaluator):
    def __init__(self, unlearn_config, test_label, test_pred, df_label, df_pred, dr_label=None, dr_pred=None, df_mask=None, tokenizer=None, metric_names=['accuracy']):
        # self.model = model
        self.unlearn_config = unlearn_config
        self.metric_names = metric_names

        self.test_label = test_label
        self.test_pred = test_pred
        self.df_label = df_label
        self.df_pred = df_pred
        self.dr_pred = dr_pred
        self.dr_label = dr_label
        self.tokenizer = tokenizer

        if dr_pred is not None:
            self.dr_pred = np.argmax(dr_pred, axis=1)

        self.metrics = {}
        self.evaluator = {}
        for metric_name in self.metric_names:
            self.evaluator[metric_name] = evaluate.load(metric_name)

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_text_gen_metrics(self, metric, labels, preds):
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: v for k, v in result.items() if 'rouge' in k}
        return result

    def compute(self):
        self.get_test_performance()
        self.get_df_performance()
        if self.dr_pred is not None:
            self.get_knowledge_gap()
            self.get_dr_performance()

        for metric_name in self.metric_names:
            if metric_name == 'rouge':
                metric_name = 'rougeL'
            self.metrics['unlearn_overall_' + metric_name] = self.metrics['test_' + metric_name] + (1 - self.metrics['df_' + metric_name])
            if self.dr_pred is not None:
                self.metrics['unlearn_overall_' + metric_name] = (self.metrics['unlearn_overall_' + metric_name] + self.metrics['knowledge_gap']) / 3
            else:
                self.metrics['unlearn_overall_' + metric_name] = self.metrics['unlearn_overall_' + metric_name] / 2

        return self.metrics

    def get_test_performance(self):
        for metric_name in self.metric_names:
            evaluator = self.evaluator[metric_name]
            metric_val = self.compute_text_gen_metrics(evaluator, self.test_label, self.test_pred)
            if metric_name == 'rouge':
                metric_name = 'rougeL'
            self.metrics['test_' + metric_name] = metric_val[metric_name]

    def get_df_performance(self):
        for metric_name in self.metric_names:
            evaluator = self.evaluator[metric_name]
            metric_val = self.compute_text_gen_metrics(evaluator, self.df_label, self.df_pred)
            if metric_name == 'rouge':
                metric_name = 'rougeL'
            self.metrics['df_' + metric_name] = metric_val[metric_name]

    def get_dr_performance(self):
        for metric_name in self.metric_names:
            evaluator = self.evaluator[metric_name]
            metric_val = self.compute_text_gen_metrics(evaluator, self.dr_label, self.dr_pred)
            if metric_name == 'rouge':
                metric_name = 'rougeL'
            self.metrics['dr_' + metric_name] = metric_val[metric_name]
