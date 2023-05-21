

import requests 
import argparse
import jsonlines
import json
from collections import defaultdict
import csv
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

def search_cord(claim, cord_doi_dict):
    #Search for documents matching any of query terms (either in title or abstract)
    search_request_any = {
        'yql': 'select id,title, abstract, id, doi from sources * where userQuery();',
        'hits': 10,
        'summary': 'short',
        'timeout': '1.0s',
        'query': claim,
        'type': 'any',
        'ranking': 'default'
    }

    # request 
    endpoint='https://api.cord19.vespa.ai/search/'
    response = requests.post(endpoint, json=search_request_any)

    return_list = set()
    retrieved_abstracts = dict()
    #try:
    root = json.loads(response.text)["root"]
    if "children" in root:
        for child in root["children"]:
            if "doi" in child["fields"]:
                doi_id = child["fields"]["doi"].replace("https://doi.org/", "")
                if doi_id in cord_doi_dict:
                    cord = cord_doi_dict[doi_id]
                    sentences = sent_tokenize(cord["abstract"])
                    if len(sentences) <= 25 and len(sentences) > 1: 
                        # <= 20 abstracts data in cord is ditry, need to filter out
                        # > 1 filter out failed to splite the abstract
                        retrieved_abstracts[cord["cord_id"]] = [cord["title"]] + sentences
                        return_list.add(cord["cord_id"])


    # except:
    #     print(response.text)
    #     exit()
    return list(return_list), retrieved_abstracts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--cord-corpus', type=str, required=False)
    parser.add_argument('--is-oracle', type=bool, default=False, required=False)
    args = parser.parse_args()

    dataset = list(jsonlines.open(args.dataset))
    #output = jsonlines.open(args.output, 'w')
    output = open(args.output,'w')

    if not args.is_oracle:
        # load cord
        cord_doi_dict = defaultdict(list)
        cord_id_dict = defaultdict(list)

        with open(args.cord_corpus) as f_in:
            reader = csv.DictReader(f_in)
            for row in tqdm(reader):
                cord_doi_dict[row['doi']] = {
                    'title': row['title'],
                    'abstract': row['abstract'],
                    'cord_id': row['cord_uid']
                }
                cord_id_dict[row['cord_id']] = {
                    'title': row['title'],
                    'abstract': row['abstract'],
                    'cord_id': row['cord_uid']
                }

    for data in tqdm(dataset):
        if args.is_oracle:
            retrieved_cord_ids = [data['cord_id']]
            retrieved_abstracts = None #[data['cord_id']] = [cord["title"]] + 
        else:
            retrieved_cord_ids, retrieved_abstracts = search_cord(data["claim"], cord_doi_dict)
  
        # output.write({
        #     'claim_id': data['id'],
        #     'doc_ids': retrieved_cord_ids,
        #     'abstracts': retrieved_abstracts
        # })
        output.write(json.dumps({
            'claim_id': data['id'],
            'doc_ids': retrieved_cord_ids,
            'abstracts': retrieved_abstracts
        })+'\n')



# #Search for documents matching all query terms (either in title or abstract)
# search_request_all = {
#   'yql': 'select id,title, abstract, doi from sources * where userQuery();',
#   'hits': 5,
#   'summary': 'short',
#   'timeout': '1.0s',
#   'query': 'coronavirus temperature sensitivity',
#   'type': 'all',
#   'ranking': 'default'
# }

# #Search for documents matching any of query terms (either in title or abstract)
# search_request_any = {
#   'yql': 'select id,title, abstract, doi from sources * where userQuery();',
#   'hits': 5,
#   'summary': 'short',
#   'timeout': '1.0s',
#   'query': 'coronavirus temperature sensitivity',
#   'type': 'any',
#   'ranking': 'default'
# }

# #Restrict matching to abstract field and filter by timestamp
# search_request_all_abstract = {
#   'yql': 'select id,title, abstract, doi from sources * where userQuery() and has_full_text=true and timestamp > 1577836800;',
#   'default-index': 'abstract',
#   'hits': 5,
#   'summary': 'short',
#   'timeout': '1.0s',
#   'query': '"sars-cov-2" temperature',
#   'type': 'all',
#   'ranking': 'default'
# }
