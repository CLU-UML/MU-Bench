import os
import evaluate
import numpy as np
import nltk
from sklearn.metrics import roc_auc_score


class Evaluator:
    def __init__(self, config, test_label, test_pred, df_label, df_pred, dr_label=None, dr_pred=None, df_mask=None, tokenizer=None, metric_names=['accuracy']):
        # self.model = model
        self.config = config
        self.metric_names = metric_names

        self.test_label = test_label
        self.test_pred = test_pred
        self.df_label = df_label
        self.df_pred = df_pred
        self.dr_pred = dr_pred
        self.dr_label = dr_label
        self.tokenizer = tokenizer

        self.test_pred = np.argmax(test_pred, axis=1)
        self.df_pred = np.argmax(df_pred, axis=1)
        if dr_pred is not None:
            self.dr_pred = np.argmax(dr_pred, axis=1)

        self.metrics = {}
        self.evaluator = {}
        for metric_name in self.metric_names:
            self.evaluator[metric_name] = evaluate.load(metric_name)

        # if os.path.exists(f'checkpoint/text/{config.data_name}_{config.seed}/pred_logit_train.pt'):
        #     self.dtrain_logit = torch.load(f'checkpoint/text/{config.data_name}_{config.seed}/pred_logit_train.pt', map_location='cpu')

        # if os.path.exists(f'checkpoint/text/{config.data_name}_{config.seed}/pred_logit_validation.pt'):
        #     self.dvalidation_logit = torch.load(f'checkpoint/text/{config.data_name}_{config.seed}/pred_logit_validation.pt', map_location='cpu')

    def compute(self):
        self.get_test_performance()
        self.get_df_performance()
        if self.dr_pred is not None:
            self.get_knowledge_gap()
            self.get_dr_performance()

        for metric_name in self.metric_names:
            self.metrics['unlearn_overall_' + metric_name] = self.metrics['test_' + metric_name] + (1 - self.metrics['df_' + metric_name])
            if self.dr_pred is not None:
                self.metrics['unlearn_overall_' + metric_name] = (self.metrics['unlearn_overall_' + metric_name] + self.metrics['knowledge_gap']) / 3
            else:
                self.metrics['unlearn_overall_' + metric_name] = self.metrics['unlearn_overall_' + metric_name] / 2

        return self.metrics

    def get_test_performance(self):
        for metric_name in self.metric_names:
            evaluator = self.evaluator[metric_name]
            metric_val = evaluator.compute(references=self.test_label, predictions=self.test_pred)
            if metric_name == 'rouge':
                metric_name = 'rougeL'
            self.metrics['test_' + metric_name] = metric_val[metric_name]

    def get_df_performance(self):
        for metric_name in self.metric_names:
            evaluator = self.evaluator[metric_name]
            metric_val = evaluator.compute(references=self.df_label, predictions=self.df_pred)
            if metric_name == 'rouge':
                metric_name = 'rougeL'
            self.metrics['df_' + metric_name] = metric_val[metric_name]

    def get_dr_performance(self):
        for metric_name in self.metric_names:
            evaluator = self.evaluator[metric_name]
            metric_val = evaluator.compute(references=self.dr_label, predictions=self.dr_pred)
            if metric_name == 'rouge':
                metric_name = 'rougeL'
            self.metrics['dr_' + metric_name] = metric_val[metric_name]

    def get_knowledge_gap(self):
        df_size = self.df_label.shape[0]
        label = [1] * df_size + [0] * df_size

        gap = []
        all_idx = np.arange(self.dr_label.shape[0])
        for _ in range(500):
            sel_idx = np.random.choice(all_idx, df_size, replace=False)
            logit = np.hstack([self.dr_pred[sel_idx], self.df_pred])
            auc = roc_auc_score(label, logit)
            gap.append(auc)
        
        self.metrics['knowledge_gap'] = np.mean(gap)



class TextGenEvaluator(Evaluator):
    def __init__(self, config, test_label, test_pred, df_label, df_pred, dr_label=None, dr_pred=None, df_mask=None, tokenizer=None, metric_names=['accuracy']):
        # self.model = model
        self.config = config
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
