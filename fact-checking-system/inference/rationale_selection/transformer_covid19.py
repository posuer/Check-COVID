import argparse
import jsonlines

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=False)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--abstract-retrieval', type=str, required=False)
parser.add_argument('--is-oracle', type=bool, required=False, default=False)
parser.add_argument('--model', type=str, required=False)
parser.add_argument('--threshold', type=float, default=0.5, required=False)
parser.add_argument('--only-rationale', action='store_true')
parser.add_argument('--output-flex', type=str)
parser.add_argument('--output-k2', type=str)
parser.add_argument('--output-k3', type=str)
parser.add_argument('--output-k4', type=str)
parser.add_argument('--output-k5', type=str)
parser.add_argument('--no-cuda', type=bool)
parser.add_argument('--no-empty-rationales', type=bool, default=False, required=False)

args = parser.parse_args()

corpus = {doc['cord_id']: doc for doc in jsonlines.open(args.corpus)}
dataset = jsonlines.open(args.dataset)
abstract_retrieval = jsonlines.open(args.abstract_retrieval)

if args.is_oracle:
    if args.output_flex:
        output_path = args.output_flex
    output = jsonlines.open(output_path, 'w')
    for data in tqdm(dataset):
        evidence = {
            data["cord_id"] : [evid["sent_index"] for evid in data["evidence_set"]]
        }
        output.write({
            'claim_id': data['id'],
            'evidence': evidence
        })
    exit()


if not args.no_cuda:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

print(f'Using device "{device}"')

tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir="cache")
model = AutoModelForSequenceClassification.from_pretrained(args.model, cache_dir="cache", return_dict=True).to(device).eval()

results = []

with torch.no_grad():
    for data, retrieval in tqdm(list(zip(dataset, abstract_retrieval))):
    #for data in tqdm(list(dataset)):
        assert data['id'] == retrieval['claim_id']
        claim = data['claim']

        evidence_scores = {}
        for doc_id in retrieval['doc_ids']:
            #doc = corpus[data["cord_id"]]
            if "abstracts" in retrieval and retrieval["abstracts"]:
                sentences = retrieval["abstracts"][doc_id]
            else:
                doc = corpus[doc_id]
                sentences = doc['abstract']
            #import pdb; pdb.set_trace()
            encoded_dict = tokenizer.batch_encode_plus(
                [list (a) for a in zip(sentences, [claim] * len(sentences))] if not args.only_rationale else sentences,
                pad_to_max_length=True,
                return_tensors='pt',
                max_length=512,
            )
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            sentence_scores = torch.softmax(model(**encoded_dict)[0], dim=1)[:, 1].detach().cpu().numpy()
            evidence_scores[doc_id] = sentence_scores

        results.append({
            'claim_id': data['id'],
            'evidence_scores': evidence_scores,
            'abstracts': retrieval["abstracts"] 
        } if "abstracts" in retrieval else
            {
            'claim_id': data['id'],
            'evidence_scores': evidence_scores
        }
        )
    
def output_k(output_path, k=None):
    output = jsonlines.open(output_path, 'w')
    for result in results:
        if k:
            evidence = {doc_id: list(sorted(sentence_scores.argsort()[-k:][::-1].tolist()))
                        for doc_id, sentence_scores in result['evidence_scores'].items()}
        else:
            evidence = {doc_id: (sentence_scores >= args.threshold).nonzero()[0].tolist()
                        for doc_id, sentence_scores in result['evidence_scores'].items()}
        if args.no_empty_rationales:
            for doc_id, sentence_scores in result['evidence_scores'].items():
                if len(evidence[doc_id]) == 0:
                    evidence[doc_id] = list(sorted(sentence_scores.argsort()[-1:][::-1].tolist()))

        output.write({
            'claim_id': result['claim_id'],
            'evidence': evidence,
            'abstracts': result["abstracts"] 
        } if "abstracts" in result else
            {
            'claim_id': result['claim_id'],
            'evidence': evidence,
        })

if args.output_flex:
    output_k(args.output_flex)

if args.output_k2:
    output_k(args.output_k2, 2)

if args.output_k3:
    output_k(args.output_k3, 3)

if args.output_k4:
    output_k(args.output_k4, 4)

if args.output_k5:
    output_k(args.output_k5, 5)