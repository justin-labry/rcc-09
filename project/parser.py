# imports
from util import list_to_string, split_into_sentences, normalize_string, label_substring, encode, encode_test, extract_formatted_data, extract_formatted_data_test
import codecs
import json
from nltk.tokenize import word_tokenize
import re
import csv
import unicodedata
from tqdm import tqdm
import pickle
import argparse
import logging
import ast
import itertools

from sklearn.model_selection import train_test_split
from collections import Counter


def get_args():
    parser = argparse.ArgumentParser(
        description='dd')
    parser.add_argument('--train', type=str, default='./formatted-data/rcc_corpus_train.csv',
                        help='train data path')
    parser.add_argument('--test', type=str, default='./formatted-data/rcc_corpus_dev_annotated.csv',
                        help='test data path')
    parser.add_argument('--test_preprocessed', type=str, default='./formatted-data/rcc_corpus_dev.csv',
                        help='test data path')
    parser.add_argument('--ratio', type=float, default=0.15,
                        help='ratio of labeled data')
    args = parser.parse_args()
    return args


def train_set_parser(publication_txt_path_prefix, publications_json_path, data_set_citations_json_path, data_sets_json_path, output_filename):
    citation_dict = dict()
    print("Loading data_set_citations.json file...")
    # open the publications.json file
    with open(data_set_citations_json_path) as json_data_set_citations:
        # parse it as JSON
        data_set_citations = json.load(json_data_set_citations)
        # loop
        for citaion_info in tqdm(data_set_citations, total=len(data_set_citations)):
            publication_id = citaion_info.get( "publication_id", None )
            data_set_id = citaion_info.get( "data_set_id", None )
            mention_list = citaion_info.get( "mention_list", None )
            formatted_mention_list = []
            for mention in mention_list:
                mention.encode('ascii','ignore')
                mention = normalize_string(mention)
                sentences = split_into_sentences(mention)
                words = []
                for sentence in sentences:
                    words += word_tokenize(sentence)
                words = [w for w in words if len(w)<15]
                if len(words) > 0:
                    formatted_mention_list.append(words)
            if publication_id in citation_dict:
                citation_dict[publication_id].append([data_set_id, formatted_mention_list])
            else:
                citation_dict[publication_id] = [[data_set_id, formatted_mention_list]]
    # set prefix to formatted publication txt files
    formatted_txt_path_prefix = "./formatted-data/"
    # set path to publications.json
    formatted_publications = dict()
    print("Tokenizing publication files...")
    # open the publications.json file
    with open(publications_json_path) as json_publication_file:
        # parse it as JSON
        publication_list = json.load(json_publication_file)
        # loop over the elements in the list
        for publication_info in tqdm(publication_list, total=len(publication_list)):
            # get information on publication:
            publication_id = publication_info.get( "publication_id", None )
            text_file_name = publication_info.get( "text_file_name", None )
            # get raw text
            raw_text = ''
            txt_file_path = publication_txt_path_prefix + text_file_name
            with open(txt_file_path) as txt_file:
                for line in txt_file:
                    stripped_line = line.strip()
                    raw_text += ' ' + stripped_line
                    if len(stripped_line.split()) <= 5:
                        raw_text += '<stop>'    # marking for sentence boundary in split_into_sentences() function
            raw_text.encode('ascii','ignore')
            raw_text = normalize_string(raw_text)
            # add to formatted_publications dictionary
            formatted_text_list = []
            sentences = split_into_sentences(raw_text)
            for sentence in sentences:
                words = word_tokenize(sentence)
                words = [w for w in words if len(w)<15]
                if len(words) >= 10 and len(words) <= 30:
                    formatted_text_list.append(words)
            formatted_publications[publication_id] = formatted_text_list
    # tag mentions in publication text and write in csv file
    output_filepath = formatted_txt_path_prefix + output_filename
    with open(output_filepath, 'w') as csvfile:
        fieldnames = [
            'publication_id',
            'sentence',
            'label_sequence']
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL)
        writer.writeheader()
        print("Tagging dataset mentions in publications...")
        output = extract_formatted_data(formatted_publications, citation_dict)
        print("Writing on new csv file...", end='')
        writer.writerows(output)
        print("DONE")


def test_set_parser(publication_txt_path_prefix, publications_json_path, data_sets_json_path, output_filename):
    data_set_mention_info = []
    pub_date_dict = dict()
    print("Loading data_sets.json file...")
    with open(data_sets_json_path) as json_data_sets:
        data_sets = json.load(json_data_sets)
        for data_set_info in tqdm(data_sets, total=len(data_sets)):
            data_set_id = data_set_info.get( "data_set_id", None )
            name = data_set_info.get( "name", None )
            name.encode('ascii','ignore')
            date = data_set_info.get( "date", None )
            date.encode('ascii','ignore')
            date = date[:10]
            if 'None' in date:
                date = '1800-01-01'
            date = int(date[:4]) * 12 * 31 + int(date[5:7]) * 31 + int(date[8:10])
            mention_list = data_set_info.get( "mention_list", None )
            formatted_mention_list = []
            name = normalize_string(name)
            name_words = word_tokenize(name)
            formatted_mention_list.append([name_words, list_to_string((name_words))])
            for mention in mention_list:
                mention.encode('ascii','ignore')
                mention = normalize_string(mention).strip()
                mention = re.sub("\s\s+", " ", mention)
                if all(c.islower() for c in mention) and len(mention.split()) <= 2:
                    continue    # to avoid pronoun mentions like 'data', 'time'
                sentences = split_into_sentences(mention)
                words = []
                for sentence in sentences:
                    words += word_tokenize(sentence)
                words = [w for w in words if len(w)<15]
                if len(words) > 0:
                    formatted_mention_list.append([words, list_to_string(words)])
            data_set_mention_info.append([date, data_set_id, formatted_mention_list])
    data_set_mention_info.sort(key=lambda x: int(x[0]), reverse=True)
    # set prefix to formatted publication txt files
    formatted_txt_path_prefix = "./formatted-data/"
    # set path to publications.json
    formatted_publications = dict()
    print("Tokenizing publications.json file...")
    # open the publications.json file
    with open(publications_json_path) as json_publication_file:
        # parse it as JSON
        publication_list = json.load(json_publication_file)
        # loop over the elements in the list
        for publication_info in tqdm(publication_list, total=len(publication_list)):
            # get information on publication:
            publication_id = publication_info.get( "publication_id", None )
            title = publication_info.get( "title", None )
            title.encode('ascii','ignore')
            text_file_name = publication_info.get( "text_file_name", None )
            unique_identifier = publication_info.get( "unique_identifier", None ) # id가 bbk로 시작하면 pub_date은 None임
            if 'bbk' not in unique_identifier:
                pub_date = publication_info.get( "pub_date", None )
            else:
                pub_date = '2200-01-01'
            pub_date.encode('ascii', 'ignore')
            pub_date = int(pub_date[:4]) * 12 * 31 + int(pub_date[5:7]) * 31 + int(pub_date[8:10])
            # get raw text
            raw_text = ''
            txt_file_path = publication_txt_path_prefix + text_file_name
            with open(txt_file_path) as txt_file:
                for line in txt_file:
                    stripped_line = line.strip()
                    raw_text += ' ' + stripped_line
                    if len(stripped_line.split()) <= 5:
                        raw_text += '<stop>'    # marking for sentence boundary in split_into_sentences() function
            raw_text.encode('ascii','ignore')
            raw_text = normalize_string(raw_text)
            # add to formatted_publications dictionary
            formatted_text_list = []
            chopped_raw_text = ''
            sentences = split_into_sentences(raw_text)
            for sentence in sentences:
                words = word_tokenize(sentence)
                words = [w for w in words if len(w)<15]
                if len(words) >= 10 and len(words) <= 30:
                    formatted_text_list.append(words)
                    chopped_raw_text += ' ' + list_to_string(words)
            formatted_publications[publication_id] = [formatted_text_list, chopped_raw_text.strip()]
            pub_date_dict[publication_id] = pub_date
    # tag mentions in publication text and write in csv file
    output_filepath = formatted_txt_path_prefix + output_filename
    with open(output_filepath, 'w') as csvfile:
        fieldnames = [
            'publication_id',
            'sentence',
            'label_sequence',
            'labeled']
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL)
        writer.writeheader()
        print("Tagging pre-found dataset mentions in publications...")
        output = extract_formatted_data_test(formatted_publications, data_set_mention_info, pub_date_dict)
        print("Writing on new csv file...", end='')
        writer.writerows(output)
        print("DONE")


def process_file(data_file, preprocessed=False):
    logging.info("loading data from " + data_file + " ...")
    sents = []
    tags = []
    ids = []
    labeleds = []
    with open(data_file) as df:
        lines = csv.reader(df)
        next(lines)
        for line in lines:
            publication_id = int(line[0])
            word_seq = ast.literal_eval(line[1])
            label_seq = ast.literal_eval(line[2])
            assert (len(word_seq) == len(label_seq))
            ids.append(publication_id)
            sents.append(word_seq)
            tags.append(label_seq)
            if preprocessed:
                labeled = line[3]
                labeleds.append(labeled)
    if not preprocessed:
        return ids, sents, tags
    else:
        return ids, sents, tags, labeleds


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M')
    args = get_args()
    # train_set = process_file(args.train)
    # test_set = process_file(args.test)
    # test_preprocessed = process_file(args.test_preprocessed, preprocessed=True)
    # '''
    # tag_counter = Counter(sum(train_set[1], []) + sum(test_set[1], []))
    # with open("./formatted-data/rcc_tagCollection".format(args.ratio), "w+", encoding='utf-8') as fp:
    #     fp.write('\n'.join(sorted(tag_counter.keys())))
    # '''
    # train_ids, dev_ids, train_sents, dev_sents, train_tags, dev_tags = \
    #    train_test_split(train_set[0], train_set[1], train_set[2], test_size=args.ratio)
    # train_set = [train_ids, train_sents, train_tags]
    # dev_set = [dev_ids, dev_sents, dev_tags]

    # tag_counter = Counter(list(itertools.chain.from_iterable(train_set[2])) +
    #                       list(itertools.chain.from_iterable(dev_set[2])) +
    #                       list(itertools.chain.from_iterable(test_set[2])))
    # print(tag_counter)
    # with open("./formatted-data/tagfile", "w", encoding='utf-8') as fp:
    #     fp.write('\n'.join(sorted(tag_counter.keys())))

    # logging.info("#train data: {}".format(len(train_set[0])))
    # logging.info("#dev data: {}".format(len(dev_set[0])))
    # logging.info("#test data: {}".format(len(test_set[0])))

    # pickle.dump(
    #     [train_set, dev_set, test_set, test_preprocessed], open("./formatted-data/rcc.data".format(args.ratio), "wb+"),
    #     protocol=-1)
    # for trainset
    # publication_txt_path_prefix = "../train-data/files/text/"
    # publications_json_path = "../train-data/publications.json"
    # data_set_citations_json_path = "../train-data/data_set_citations.json"
    # data_sets_json_path = '../train-data/data_sets.json'
    # output_filename = 'rcc_corpus_train.csv'
    # print("Preparing trainset")
    # train_set_parser(publication_txt_path_prefix, publications_json_path, data_set_citations_json_path, data_sets_json_path, output_filename)
    # # for devset
    # publication_txt_path_prefix = "./dev_data/text/"
    # publications_json_path = "./dev_data/publications.json"
    # data_set_citations_json_path = "./dev_data/data_set_citations.json"
    # output_filename = 'rcc_corpus_dev_annotated.csv'
    # print("Preparing devset")
    # train_set_parser(publication_txt_path_prefix, publications_json_path, data_set_citations_json_path, output_filename)
    # for preprocessing devset
    publication_txt_path_prefix = "../train-data/files/text/"
    publications_json_path = "../train-data/publications.json"
    data_sets_json_path = '../train-data/data_sets.json'
    output_filename = 'rcc_corpus_train_by_bruteforce.csv'
    test_set_parser(publication_txt_path_prefix, publications_json_path, data_sets_json_path, output_filename)
