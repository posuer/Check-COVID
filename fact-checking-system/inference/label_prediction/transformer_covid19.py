import argparse
import jsonlines

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

#from verisci.inference.label_prediction.specificity import get_specificity

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=False)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--rationale-selection', type=str, required=True)
parser.add_argument('--mode', type=str, default='claim_and_rationale', choices=['claim_and_rationale', 'only_claim', 'only_rationale'])
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--include_nouns', type=bool, default=False )
parser.add_argument('--fuzzy', type=bool, default=False)
parser.add_argument('--specificity_thres', type=float, default=None)
parser.add_argument('--no_nei_pred', type=bool, default=False, required=False) # Do not allow NEI predictions

args = parser.parse_args()

print(args.mode)

corpus = {doc['cord_id']: doc for doc in jsonlines.open(args.corpus)}
dataset = jsonlines.open(args.dataset)
rationale_selection = jsonlines.open(args.rationale_selection)
output = jsonlines.open(args.output, 'w')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device "{device}"')

tokenizer = AutoTokenizer.from_pretrained(args.model)
config = AutoConfig.from_pretrained(args.model, num_labels=3)
model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config).eval().to(device)

LABELS = ['REFUTE', 'NOTENOUGHINFO', 'SUPPORT']

def encode(sentences, claims):
    text = {
        "claim_and_rationale": list(zip(sentences, claims)),
        "only_claim": claims,
        "only_rationale": sentences
    }[args.mode]
    encoded_dict = tokenizer.batch_encode_plus(
        text,
        pad_to_max_length=True,
        return_tensors='pt'
    )

    if encoded_dict['input_ids'].size(1) > 512:
        encoded_dict = tokenizer.batch_encode_plus(
            text,
            max_length=512,
            pad_to_max_length=True,
            truncation_strategy='only_first',
            return_tensors='pt'
        )
    encoded_dict = {key: tensor.to(device)
                  for key, tensor in encoded_dict.items()}
    return encoded_dict


with torch.no_grad():
    for data, selection in tqdm(list(zip(dataset, rationale_selection))):
        assert data['id'] == selection['claim_id']

        claim = data['claim']
        results = {}

        # if not selection['evidence']: # deal with non-claim
        #     results["000"] = {'label': 'NOTENOUGHINFO', 'confidence': 1}

        for doc_id, indices in selection['evidence'].items():
            if not indices:
                results[doc_id] = {'label': 'NOTENOUGHINFO', 'confidence': 1}
            else:
                if "abstracts" in selection and selection["abstracts"]:
                    evidence = ' '.join([selection["abstracts"][doc_id][i] for i in indices])
                else:
                    evidence = ' '.join([corpus[doc_id]['abstract'][i] for i in indices])
                encoded_dict = encode([evidence], [claim])
                label_scores = torch.softmax(model(**encoded_dict)[0], dim=1)[0]
                label_index = label_scores.argmax().item()
                if args.no_nei_pred:
                    label_index = 0
                    if label_scores[0].item() < label_scores[2].item():
                        label_index = 2
                label_confidence = label_scores[label_index].item()
                results[doc_id] = {'label': LABELS[label_index], 'confidence': round(label_confidence, 4)}

        output.write({
            'claim_id': data['id'],
            'labels': results
        })