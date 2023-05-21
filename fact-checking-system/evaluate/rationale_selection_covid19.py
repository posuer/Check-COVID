"""
Computes the rationale selection score as in the paper.
"""


import argparse
import jsonlines
from collections import Counter
import os
from lib.metrics import compute_f1


def is_sentence_correct(pred_sentence, pred_sentences, gold_sets, version):
    """
    Standard: A predicted sentence is correctly identified if it is part of a gold
    rationale.
    Global Recall: A predicted sentence is correctly identified if it is part of a gold
    rationale, and all other sentences in the gold rationale are also
    predicted rationale sentences.
    """
    if pred_sentence in gold_sets:
        if version == 'standard':
            return True
        elif version == 'global_recall' and all([x in pred_sentences for x in gold_sets]): 
            return True
    return False

def is_rationale_correct(pred_sentences, gold_sets):
    """
    returns True if the entirety of the predicted rationale matches 
    the entirety of the gold rationale.
    """
    if pred_sentences == gold_sets:
        return True
    return False

def is_intersection(pred_sentences, gold_sets):
    """
    returns True if any of the predicted sentences in the predicted rationale 
    are in the gold rationale.
    """
    for pred_sent in pred_sentences:
        if pred_sent in gold_sets:
            return True
    return False



parser = argparse.ArgumentParser()
#parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--rationale-selection', type=str, required=True)
parser.add_argument('--deleting-model-path', type=str, default=None, required=False)
parser.add_argument('--deleting-model-threshold', type=float, default=0.0, required=False)
parser.add_argument('--rationale-exact-match', type=bool, required=False, default=False)
parser.add_argument('--rationale-intersection', type=bool, required=False, default=False)
parser.add_argument('--sentence-score-version', type=str, default='global_recall', required=False)


args = parser.parse_args()


#corpus = {doc['cord_id']: doc for doc in jsonlines.open(args.corpus)}
dataset = jsonlines.open(args.dataset)
rationale_selection = jsonlines.open(args.rationale_selection)

counts = Counter()


for data, retrieval in zip(dataset, rationale_selection):
    assert data['id'] == retrieval['claim_id']
    if data['label'] == "NOTENOUGHINFO":
        continue

    # Count all the gold evidence sentences.
    # for doc_key, gold_rationales in data["evidence"].items():
    #     for entry in gold_rationales:
    #         counts["relevant"] += len(entry["sentences"])
    
    if args.rationale_exact_match == True or args.rationale_intersection == True: 
        counts["relevant"] += 1
    else:
        counts["relevant"] += len(data["evidence_set"])

    claim_id = retrieval['claim_id']

    for doc_id, pred_sentences in retrieval['evidence'].items():
        #true_evidence_sets = data['evidence'].get(doc_id) or []
        true_evidence_sets = [evd_id_list["sent_index"] for evd_id_list in data['evidence_set']]
        if args.rationale_exact_match == True: 
            counts["retrieved"] += 1
            if is_rationale_correct(pred_sentences, true_evidence_sets):
                counts["correct"] += 1
        elif args.rationale_intersection == True:
            counts["retrieved"] += 1
            if is_intersection(pred_sentences, true_evidence_sets):
                counts["correct"] += 1
        else:
            for pred_sentence in pred_sentences:
                counts["retrieved"] += 1
                if is_sentence_correct(pred_sentence, pred_sentences, true_evidence_sets, args.sentence_score_version):
                    counts["correct"] += 1

f1 = compute_f1(counts)
print(f1['precision'], f1['recall'], f1['f1'])
if args.rationale_exact_match == True or args.rationale_intersection == True:
    print(counts["correct"]/counts["relevant"])


#remove the evaluated model for space saving
if args.deleting_model_path and f1["f1"] < args.deleting_model_threshold:
    if os.path.exists(args.deleting_model_path):
        for root, subdirs, files in os.walk(args.deleting_model_path):
            for file in files:
                os.remove(os.path.join(root, file))
        os.rmdir(args.deleting_model_path)