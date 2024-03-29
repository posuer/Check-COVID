import argparse
import torch
import jsonlines
import random
import os

from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, required=True, help='Path to processed train file')
parser.add_argument('--dev', type=str, required=True, help='Path to processed train file')
parser.add_argument('--dest', type=str, required=True, help='Folder to save the weights')
parser.add_argument('--model', type=str, default='roberta-large')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch-size-gpu', type=int, default=8, help='The batch size to send through GPU')
parser.add_argument('--batch-size-accumulated', type=int, default=256, help='The batch size for each gradient update')
parser.add_argument('--lr-base', type=float, default=1e-5)
parser.add_argument('--lr-linear', type=float, default=1e-4)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device "{device}"')


class FeverLabelPredictionDataset(Dataset):
    def __init__(self, file):
        self.samples = []
        labels = {'SUPPORTS': 2, 'NOT ENOUGH INFO': 1, 'REFUTES': 0}
        for data in jsonlines.open(file):
            if data['label'] == 'NOT ENOUGH INFO':
                if data['sentences']:
                    indices = sorted(random.sample(range(len(data['sentences'])), k=1))
                    sentences = [data['sentences'][i] for i in indices]
                    self.samples.append({
                        'claim': data['claim'],
                        'rationale': ' '.join(sentences),
                        'label': labels['NOT ENOUGH INFO']
                    })
            else:
                for evidence_set in data['evidence_sets']:
                    self.samples.append({
                        'claim': data['claim'],
                        'rationale': ' '.join([data['sentences'][i] for i in evidence_set]),
                        'label': labels[data['label']]
                    })
                # Add negative samples
                non_evidence_indices = set(range(len(data['sentences']))) - set(
                    s for es in data['evidence_sets'] for s in es)
                if non_evidence_indices:
                    non_evidence_indices = random.sample(non_evidence_indices,
                                                         k=random.randint(1, min(1, len(non_evidence_indices))))
                    sentences = [data['sentences'][i] for i in non_evidence_indices]
                    self.samples.append({
                        'claim': data['claim'],
                        'rationale': ' '.join(sentences),
                        'label': labels['NOT ENOUGH INFO']
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


trainset = FeverLabelPredictionDataset(args.train)
devset = FeverLabelPredictionDataset(args.dev)

tokenizer = AutoTokenizer.from_pretrained(args.model)
config = AutoConfig.from_pretrained(args.model, num_labels=3)
model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config).to(device)
optimizer = torch.optim.Adam([
    # If you are using non-roberta based models, change this to point to the right base
    {'params': model.roberta.parameters(), 'lr': args.lr_base},
    {'params': model.classifier.parameters(), 'lr': args.lr_linear}
])
scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 20)


def encode(claims: List[str], rationale: List[str]):
    encoded_dict = tokenizer.batch_encode_plus(
        zip(rationale, claims),
        pad_to_max_length=True,
        return_tensors='pt')
    if encoded_dict['input_ids'].size(1) > 512:
        # Too long for the model. Truncate it
        encoded_dict = tokenizer.batch_encode_plus(
            zip(rationale, claims),
            max_length=512,
            truncation_strategy='only_first',
            pad_to_max_length=True,
            return_tensors='pt')
    encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
    return encoded_dict


def evaluate(model, dataset):
    model.eval()
    targets = []
    outputs = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=args.batch_size_gpu):
            encoded_dict = encode(batch['claim'], batch['rationale'])
            logits = model(**encoded_dict)[0]
            targets.extend(batch['label'].float().tolist())
            outputs.extend(logits.argmax(dim=1).tolist())
    return {
        'macro_f1': f1_score(targets, outputs, zero_division=0, average='macro'),
        'f1': tuple(f1_score(targets, outputs, zero_division=0, average=None)),
        'precision': tuple(precision_score(targets, outputs, zero_division=0, average=None)),
        'recall': tuple(recall_score(targets, outputs, zero_division=0, average=None))
    }


for e in range(args.epochs):
    model.train()
    t = tqdm(DataLoader(trainset, batch_size=args.batch_size_gpu, shuffle=True))
    for i, batch in enumerate(t):
        encoded_dict = encode(batch['claim'], batch['rationale'])
        loss, logits = model(**encoded_dict, labels=batch['label'].long().to(device))
        loss.backward()
        if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
            optimizer.step()
            optimizer.zero_grad()
            t.set_description(f'Epoch {e}, iter {i}, loss: {round(loss.item(), 4)}')
    scheduler.step()
    # Eval
    train_score = evaluate(model, trainset)
    print(f'Epoch {e} train score:')
    print(train_score)
    dev_score = evaluate(model, devset)
    print(f'Epoch {e} dev score:')
    print(dev_score)
    # Save
    save_path = os.path.join(args.dest, f'epoch-{e}-f1-{int(dev_score["macro_f1"] * 1e4)}')
    os.makedirs(save_path)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
