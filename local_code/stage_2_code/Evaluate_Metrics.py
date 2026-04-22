'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score


from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np


class Evaluate_Metrics(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...')

        true_y = np.array(self.data['true_y'])
        pred_y = np.array(self.data['pred_y'])

        metrics = {
            'accuracy': accuracy_score(true_y, pred_y),
            'precision_macro': precision_score(true_y, pred_y, average='macro', zero_division=0),
            'recall_macro': recall_score(true_y, pred_y, average='macro', zero_division=0),
            'f1_macro': f1_score(true_y, pred_y, average='macro', zero_division=0),
            'precision_weighted': precision_score(true_y, pred_y, average='weighted', zero_division=0),
            'recall_weighted': recall_score(true_y, pred_y, average='weighted', zero_division=0),
            'f1_weighted': f1_score(true_y, pred_y, average='weighted', zero_division=0),
            'classification_report': classification_report(true_y, pred_y, zero_division=0)
        }

        print('Accuracy:', metrics['accuracy'])
        print('Macro Precision:', metrics['precision_macro'])
        print('Macro Recall:', metrics['recall_macro'])
        print('Macro F1:', metrics['f1_macro'])
        print('Weighted Precision:', metrics['precision_weighted'])
        print('Weighted Recall:', metrics['recall_weighted'])
        print('Weighted F1:', metrics['f1_weighted'])
        print('\nClassification Report:\n')
        print(metrics['classification_report'])

        return metrics
        