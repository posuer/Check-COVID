import os
import argparse
import jsonlines

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=False)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--label-prediction', type=str, required=True)
parser.add_argument('--filter', type=str, choices=['structured', 'unstructured'])
parser.add_argument('--deleting-model-path', type=str, default=None, required=False)
parser.add_argument('--deleting-model-macroF1-threshold', type=float, default=0.0, required=False)

args = parser.parse_args()

#corpus = {doc['cord_id']: doc for doc in jsonlines.open(args.corpus)}
dataset = jsonlines.open(args.dataset)
label_prediction = jsonlines.open(args.label_prediction)

pred_labels = []
true_labels = []

LABELS = {'REFUTE': 0, 'NOTENOUGHINFO': 1, 'SUPPORT': 2}

for data, prediction in zip(dataset, label_prediction):
    assert data['id'] == prediction['claim_id']

    if args.filter:
        prediction['labels'] = {doc_id: pred for doc_id, pred in prediction['labels'].items()
                                if corpus[doc_id]['structured'] is (args.filter == 'structured')}
    if not prediction['labels']:
        continue

    claim_id = data['id']
    for doc_id, pred in prediction['labels'].items():
        pred_label = pred['label']
        true_label = [data['label']]
        assert len(true_label) <= 1, 'Currently support only one label per doc'
        true_label = next(iter(true_label)) if true_label else 'NOTENOUGHINFO'
        pred_labels.append(LABELS[pred_label])
        true_labels.append(LABELS[true_label])

print(f'Accuracy           {round(sum([pred_labels[i] == true_labels[i] for i in range(len(pred_labels))]) / len(pred_labels), 4)}')
print(f'Macro F1:          {f1_score(true_labels, pred_labels, average="macro").round(4)}')
print(f'Macro F1 w/o NEI:  {f1_score(true_labels, pred_labels, average="macro", labels=[0, 2]).round(4)}')
print()
print('                   [Refute  NotEnIn  Support ]')
print(f'F1:                {f1_score(true_labels, pred_labels, average=None).round(4)}')
print(f'Precision:         {precision_score(true_labels, pred_labels, average=None).round(4)}')
print(f'Recall:            {recall_score(true_labels, pred_labels, average=None).round(4)}')
print()
print('Confusion Matrix:')
print(confusion_matrix(true_labels, pred_labels))


#remove the evaluated model for space saving
#  sum([pred_labels[i] == true_labels[i] for i in range(len(pred_labels))]) / len(pred_labels) < 0.72 and
if args.deleting_model_path and f1_score(true_labels, pred_labels, average="macro") < args.deleting_model_macroF1_threshold:
    if os.path.exists(args.deleting_model_path):
        for root, subdirs, files in os.walk(args.deleting_model_path):
            for file in files:
                os.remove(os.path.join(root, file))
        os.rmdir(args.deleting_model_path)
