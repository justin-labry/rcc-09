#from util import list_to_string
import pickle
import logging
import argparse
import logging
import pickle
import csv
import ast
import json

def list_to_string(list_tokens):
    res = ''
    first = True
    for tok in list_tokens:
        if first:
            first = False
        else:
            res += ' '
        res += tok
    return res


def get_args():
    parser = argparse.ArgumentParser(description='rcc-09')
    parser.add_argument('--test_preprocessed', type=str, default='./formatted-data/rcc_corpus_test.csv',
                        help='processed test data path')
    parser.add_argument('--unordered_output_path', type=str, default='./formatted-data/model_output_mentions.csv')
    #dataset_citations
    parser.add_argument('--dataset_citations_path', type=str, default='../data/output/data_set_citations.json')
    parser.add_argument('--data_set_mentions_path', type=str, default='../data/output/data_set_mentions.json')
    args = parser.parse_args()
    return args

args = get_args()
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
logging.info("test data(preprocessed): {}".format(args.test_preprocessed))
test_annotated = []
with open(args.test_preprocessed) as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        publication_id = int(line[0])
        word_seq = ast.literal_eval(line[1])
        label_seq = ast.literal_eval(line[2])
        assert (len(word_seq) == len(label_seq))
        found = False
        mentions_words = []
        dataset_id = 0
        for w, l in zip(word_seq, label_seq):
            if 'B' in l:
                mentions_words.append(w)
                found = True
                dataset_id = l[2:]
            elif l == 'I':
                mentions_words.append(w)
            else:
                if found:
                    test_annotated.append([mentions_words, publication_id, dataset_id, 1.0])
                    found = False
                    mentions_words = []
        if found:
            test_annotated.append([mentions_words, publication_id, dataset_id, 1.0])

with open(args.unordered_output_path) as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        mentions_words = ast.literal_eval(line[0])
        publication_id = int(line[1])
        dataset_id = int(line[2])
        score = float(line[3])
        test_annotated.append([mentions_words, publication_id, dataset_id, score])

dataset_citations_mentionlist_dict = dict()
dataset_citations_scores_dict = dict()
mentions_dict = dict()
for mention_words, pub_id, dataset_id, score in test_annotated:
    key = (pub_id, int(dataset_id))
    key2 = (pub_id, list_to_string(mention_words))
    if key in dataset_citations_mentionlist_dict:
        dataset_citations_mentionlist_dict[key].append(list_to_string(mention_words))
        dataset_citations_scores_dict[key] = [dataset_citations_scores_dict[key][0] + score,
                                              dataset_citations_scores_dict[key][1] + 1]
    else:
        dataset_citations_mentionlist_dict[key] = [list_to_string(mention_words)]
        dataset_citations_scores_dict[key] = [score, 1]
    if key2 in mentions_dict:
        mentions_dict[key2] = [mentions_dict[key2][0] + score,
                               mentions_dict[key2][1] + 1]
    else:
        mentions_dict[key2] = [score, 1]


dataset_citations = []
for key, mention_list in dataset_citations_mentionlist_dict.items():
    citation_dict = dict()
    score = dataset_citations_scores_dict[key][0] / dataset_citations_scores_dict[key][1]
    pub_id = key[0]
    dataset_id = key[1]
    citation_dict['publication_id'] = pub_id
    citation_dict['data_set_id'] = dataset_id
    citation_dict['mention_list'] = list(set(mention_list))
    citation_dict['score'] = round(score, 3)
    dataset_citations.append(citation_dict)

with open(args.dataset_citations_path, 'w') as outfile:
    json.dump(dataset_citations, outfile, indent=4)

dataset_mentions = []
for key, score_info in mentions_dict.items():
    men_dict = dict()
    pub_id = key[0]
    mention = key[1]
    score = mentions_dict[key][0] / mentions_dict[key][1]
    men_dict['publication_id'] = pub_id
    men_dict['mention'] = mention
    men_dict['score'] = round(score, 3)
    dataset_mentions.append(men_dict)

with open(args.data_set_mentions_path, 'w') as outfile:
    json.dump(dataset_mentions, outfile, indent=4)











