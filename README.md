# Check-COVID

## Abstract
We present a new fact-checking benchmark,
Check-COVID, that requires systems to verify
claims about COVID-19 from news using evidence from scientific articles. This approach
to fact-checking is particularly challenging as it requires checking internet text written in everyday language against evidence from journal articles written in formal academic language. Check-COVID contains 1,504 expert-annotated news claims about the coronavirus
paired with sentence-level evidence from scientific journal articles and veracity labels. It
includes both extracted (journalist-written) and
composed (annotator-written) claims. Experiments using both a fact-checking specific system (which achieved F1
scores of 76.99 and 69.90 respectively) and GPT-3.5 on this task reveal the difficulty of automatically fact-checking
both claim types and the importance of in-domain data for good performance. Our data
and models will be released publicly at [URL].
#

Disclaimer: All training, inference and eval code is built off of the code from [SciFact](https://github.com/allenai/scifact).

#
## Data
All data is included in the `CheckCOVID` directory

- `auth` are claims drawn directly from news articles (files without this specifier in the title have claims written by annotators)

- `writtenAuth` are claims written by annotators and claims drawn from news articles

- `2_classes` does not include NEI examples

- `abstracts` directory includes abstracts used to evaluate the rationale and full pipeline modules

#
## Training
### Rationale Selection Example
```
python3 fact-checking-system/training/rationale_selection/transformer_covid19_uda.py  \
    --corpus ../CheckCOVID/corpus.json  \
    --claim-train ../CheckCOVID/covidCheck_train_writtenAuth_2_classes.json  \
    --claim-dev ../CheckCOVID/covidCheck_dev_writtenAuth_2_classes.json  \
    --model models/rationale_roberta_large_fever \ # path to base rationale  FEVER model
    --dest rationale_models/fever_covid_extracted_composed  \
    --batch-size-gpu 16 --seed 42 
```
### Label Prediction Example
```
- python3 fact-checking-system/training/label_prediction/transformer_covid19_uda.py \
    --corpus ../CheckCOVID/corpus.json \
    --train ../CheckCOVID/covidCheck_train_auth.json \
    --dev ../CheckCOVID/covidCheck_dev_auth.json \
    --model <path to base label FEVER/SciFact model> \
    --dest label_models/fever_scifact_covid_extracted \
    --batch-size-gpu 16 --seed 42 \
    --batch-size-accumulated 16
```


## Inference
### Rationale Selection Example
```
python -m fact-checking-system.inference.rationale_selection.transformer_covid19 \
    --corpus ../CheckCOVID/corpus.json \
    --dataset ../CheckCOVID/covidCheck_dev_auth_2_classes.json \
    --abstract-retrieval abstracts/abstract_retrieval_oracle_dev_auth_2_classes.jsonl \ # path to abstracts
    --model rationale_models/fever_scifact_covid_extracted/epoch-6-f1-7990 \ # path to trained model
    --output-flex fever_scifact_covid_extracted.jsonl
```

### Label Prediction Example
```
python3 fact-checking-system/inference/label_prediction/transformer_covid19.py \
    --corpus ../CheckCOVID/corpus.json \
    --dataset ../CheckCOVID/covidCheck_dev_auth.json \
    --rationale-selection rationale_roberta_fever_covid19_finetuned_oracle_true_auth.jsonl \ # path to oracle rationales or predicted rationales
    --mode claim_and_rationale 
    --model label_models/fever_covid_extracted \ # path to trained model
    --output fever_covid_extracted.jsonl
```

## Scoring
### Rationale Scoring Examples
```
python3 fact-checking-system/evaluate/rationale_selection_covid19.py \
    --dataset ../CheckCOVID/covidCheck_dev_auth_2_classes.json \
    --rationale-intersection True \
    --rationale-selection rationales/fever_scifact_covid_extracted.jsonl
```
```
python3 fact-checking-system/evaluate/rationale_selection_covid19.py \
    --dataset ../CheckCOVID/covidCheck_dev_auth_2_classes.json \
    --rationale-exact-match True \
    --rationale-selection rationales/fever_scifact_covid_extracted.jsonl
```
```
python3 fact-checking-system/evaluate/rationale_selection_covid19.py \
    --dataset ../CheckCOVID/covidCheck_dev_auth_2_classes.json \
    --sentence-score-version global_recall \
    --rationale-selection rationales/fever_scifact_covid_extracted.jsonl
```
```
python3 fact-checking-system/evaluate/rationale_selection_covid19.py \
    --dataset ../CheckCOVID/covidCheck_dev_auth_2_classes.json \
    --sentence-score-version standard \
    --rationale-selection rationales/fever_scifact_covid_extracted.jsonl
```
### Label Scoring Example
```
- python3 fact-checking-system/evaluate/label_prediction_covid19.py \
    --dataset ../COVID-check/covidCheck_dev.json \
    --label-prediction ../scifact/paper_labels/covid_composed.jsonl
```


