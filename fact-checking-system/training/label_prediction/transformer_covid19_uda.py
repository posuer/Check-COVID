import argparse
import torch
import jsonlines
import random
import os
import logging

from itertools import cycle
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--train', type=str, required=True, help='Path to processed train file')
parser.add_argument('--dev', type=str, required=True, help='Path to processed train file')
parser.add_argument('--dest', type=str, required=True, help='Folder to save the weights')
parser.add_argument('--model', type=str, default='roberta-large')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch-size-gpu', type=int, default=8, help='The batch size to send through GPU')
parser.add_argument('--batch-size-accumulated', type=int, default=256, help='The batch size for each gradient update')
parser.add_argument('--lr-base', type=float, default=1e-5)
parser.add_argument('--lr-linear', type=float, default=1e-4)

parser.add_argument('--data-uda', type=str, required=False, help='raw and augmented unlabeled data for UDA')
parser.add_argument('--batch-size-unsup-ratio', type=float, default=0, help='The batch size ratio between super and unsuper')
parser.add_argument('--uda-coeff', type=float, default=1, help=' ')
parser.add_argument('--tsa', type=str, default="", choices=["", "linear_schedule", "log_schedule", "exp_schedule"], help=' Training Signal Annealing (TSA) ')
parser.add_argument('--uda-softmax-temp', type=float, default=-1, help="Sharpening Predictions 0.4")
parser.add_argument('--uda-confidence-thresh', type=float, default=-1, help='Confidence-based masking 0.8')
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

args = parser.parse_args()

# Setup logging
logger = logging.getLogger(__name__)
if not os.path.exists(args.dest):
    os.makedirs(args.dest)
logging.basicConfig(
    filename=os.path.join(args.dest,"logging.txt"),
    filemode='w',
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
if args.seed:
    torch.manual_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device "{device}"')
logger.info(args)

class FeverLabelPredictionDataset(Dataset):
    def __init__(self, file):
        self.samples = []
        corpus = {doc['cord_id']: doc for doc in jsonlines.open(args.corpus)}

        labels = {'SUPPORT': 2, 'NOTENOUGHINFO': 1, 'REFUTE': 0}
        for data in jsonlines.open(file):
            # all evidences (includes "not engough info") from 'evidence_sets
            #for evidence_set in data['evidence_sets']:
            # self.samples.extend([{
            #     'claim': data['claim'],
            #     'rationale': data['sent'] in data['evidence_set'],
            #     'label': labels[data['label']]
            # } for i in evidence_set])

            rationale_idx = {s["sent_index"] for s in data['evidence_set']}
            rationale_sentences = [corpus[data["cord_id"]]['abstract'][i].strip() for i in sorted(list(rationale_idx))]
            self.samples.append({
                'claim': data['claim'],
                'rationale': ' '.join(rationale_sentences),
                'label': labels[data["label"]]  # directly use the first evidence set label
                # because currently all evidence sets have
                # the same label
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class FeverLabelPredictionDataset_UDA(Dataset):
    def __init__(self, file, is_aug=False):
        self.samples = []

        #labels = {'SUPPORTS': 2, 'NOT ENOUGH INFO': 1, 'REFUTES': 0}
        for data in jsonlines.open(file):
            #if data['label'] == 'NOT ENOUGH INFO':
            if data['abstracts']:
  
                for mid_language, back_translated_claim in data["back_translated"].items():
                
                    for abstract in data['abstracts'].values():
                        if len(abstract) > 1: #skip exampels that failed to tokenize sentence
                            self.samples.extend([{
                                'claim': back_translated_claim if is_aug else data['claim'] ,
                                'rationale': sent,
                                #'label': labels['NOT ENOUGH INFO']
                            } for sent in abstract])
                   
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]



class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)


trainset = FeverLabelPredictionDataset(args.train)
devset = FeverLabelPredictionDataset(args.dev)
if args.batch_size_unsup_ratio:
    unsupset = FeverLabelPredictionDataset_UDA(args.data_uda, is_aug=False)#[int(len(trainset)*.7):]
    augset = FeverLabelPredictionDataset_UDA(args.data_uda, is_aug=True)#[int(len(trainset)*.7):]
    assert len(unsupset) == len(augset)
    concatset = ConcatDataset(unsupset, augset)
    batch_size_unsup = int(args.batch_size_gpu * args.batch_size_unsup_ratio)


tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir="cache")
config = AutoConfig.from_pretrained(args.model, num_labels=3, cache_dir="cache")
model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config, cache_dir="cache").to(device)
optimizer = torch.optim.Adam([
    # If you are using non-roberta based models, change this to point to the right base
    {'params': model.roberta.parameters(), 'lr': args.lr_base},
    {'params': model.classifier.parameters(), 'lr': args.lr_linear}
])
scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 20)
#scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=400, num_training_steps=len(trainset)/arg.batch_size_accumulated * arg.epochs)

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
            truncation=True,
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

def unsup_feedforward(model, batch_unsup, batch_aug):
    logSoftmax_fct = torch.nn.LogSoftmax(dim=-1)

    encoded_dict_unsup = encode(batch_unsup['claim'], batch_unsup['rationale'])
    encoded_dict_aug = encode(batch_aug['claim'], batch_aug['rationale'])
    logits_u = model(**encoded_dict_unsup)[0]
    logits_aug = model(**encoded_dict_aug)[0]
    #logits_u = logSoftmax_fct(logits_u[0])
    logits_aug = logSoftmax_fct(logits_aug)

    if args.uda_softmax_temp != -1:
        #logits_u = logits_u / args.uda_softmax_temp
        logits_u = logSoftmax_fct(logits_u / args.uda_softmax_temp)

    if args.uda_confidence_thresh != -1:
        softmax_logits_u = torch.nn.Softmax(dim=-1)(logits_u)
        max_logits_u = torch.max(softmax_logits_u, -1)[0] #return is tuple (values, indices)
        unsup_loss_mask = (max_logits_u > args.uda_confidence_thresh).float().detach()
        loss_unsup = torch.nn.KLDivLoss(reduction="none")(logits_aug, logits_u.detach())[:,0] * unsup_loss_mask
        loss_unsup = torch.mean(loss_unsup) # average on all or only non-zero examples
    else:
        loss_unsup = torch.nn.KLDivLoss(reduction="batchmean")(logits_aug, logits_u.detach()) # (input, target) KLDivLoss will apply logSoftmax on target
    return loss_unsup

def get_tsa_threshold(schedule, global_step, num_train_steps, start, end):
    training_progress = float(global_step) / float(num_train_steps)
    if schedule == "linear_schedule":
        threshold = training_progress
    elif schedule == "exp_schedule":
        scale = 5
        threshold = torch.exp((torch.tensor(training_progress) - 1) * scale)
        # [exp(-5), exp(0)] = [1e-2, 1]
    elif schedule == "log_schedule":
        scale = 5
        # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
        threshold = 1 - torch.exp((torch.tensor(-training_progress)) * scale)
    return threshold * (end - start) + start


best_epoch = 0
best_dev_mF1 = 0
best_dir = ""

for e in range(args.epochs):
    model.train()
    t = tqdm(DataLoader(trainset, batch_size=args.batch_size_gpu, shuffle=False))
    if args.batch_size_unsup_ratio:
        concat_loader = DataLoader(concatset, batch_size=batch_size_unsup, shuffle=False) 
        concat_repeated = cycle(concat_loader)
    for i, batch in enumerate(t):
        encoded_dict = encode(batch['claim'], batch['rationale'])
        loss, logits = model(**encoded_dict, labels=batch['label'].long().to(device))

        if args.tsa:
            num_labels = model.num_labels#model.module.num_labels if args.n_gpu > 1 else 
            labels = batch['label'].long().to(device)
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_labels)
            
            log_probs = torch.nn.LogSoftmax(dim=-1)(logits)
            #per_example_loss = -torch.sum(one_hot_labels * log_probs, dim=-1)
            correct_label_probs = torch.sum(one_hot_labels * torch.exp(log_probs), dim=-1)

            tsa_start = 1. / num_labels
            tsa_threshold = get_tsa_threshold(args.tsa, i, int(len(trainset)/args.batch_size_gpu), tsa_start, end=1)
            larger_than_threshold = (correct_label_probs > tsa_threshold).float()
            #loss_mask = torch.ones_like(per_example_loss, dtype=torch.float)
            loss_mask = (1 - larger_than_threshold).detach()
            
            per_example_loss = torch.nn.CrossEntropyLoss(reduction='none')(logits.view(-1, num_labels), labels.view(-1))
            per_example_loss = per_example_loss * loss_mask
            loss = torch.sum(per_example_loss) / max(torch.sum(loss_mask), torch.tensor(1).cuda())
        else:
            tsa_threshold = 1
        
        if args.batch_size_unsup_ratio:
            batch_unsup, batch_aug = next(concat_repeated)
            loss_unsup = unsup_feedforward(model, batch_unsup, batch_aug)
            loss = loss + loss_unsup

        loss.backward()
        if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
            optimizer.step()
            optimizer.zero_grad()
            t.set_description(f'Epoch {e}, iter {i}, loss: {round(loss.item(), 4)}')
    scheduler.step()
    # Eval
    train_score = evaluate(model, trainset)
    logger.info(f'Epoch {e} train score:')
    logger.info(train_score)
    dev_score = evaluate(model, devset)
    logger.info(f'Epoch {e} dev score:')
    logger.info(dev_score)
    
    # Save
    if dev_score["macro_f1"] > best_dev_mF1:
        # save new checkpoint
        save_path = os.path.join(args.dest, f'epoch-{e}-f1-{int(dev_score["macro_f1"] * 1e4)}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)

        # Remove old best checkpoint
        if os.path.exists(best_dir):
            for root, subdirs, files in os.walk(best_dir):
                for file in files:
                    os.remove(os.path.join(root, file))
            os.rmdir(best_dir)

        best_epoch = e
        best_dev_mF1 = dev_score["macro_f1"]
        best_dir = save_path
        with open(os.path.join(args.dest, "best_model_path.txt"), "w") as f:
            f.write(best_dir)
    
    
    if e == args.epochs-1 or e-best_epoch>=6: #early stop
        # save last checkpoint
        save_path = os.path.join(args.dest, f'epoch-{e}-f1-{int(dev_score["macro_f1"] * 1e4)}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # tokenizer.save_pretrained(save_path)
        # model.save_pretrained(save_path)
        break

    