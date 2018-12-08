from util import list_to_string, split_into_sentences, normalize_string, label_substring, encode, encode_test, extract_formatted_data, extract_formatted_data_test
import codecs
import json
import nltk
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

nltk.download('popular')

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--publication_txt_path_prefix', type=str, default='../data/input/files/text/',
                        help='train data path')
    parser.add_argument('--publications_json_path', type=str, default='../data/input/publications.json',
                        help='test data path')
    parser.add_argument('--data_sets_json_path', type=str, default='./formatted-data/data_sets.json',
                        help='test data path')
    parser.add_argument('--output_filename', type=str, default='rcc_corpus_test.csv',
                        help='ratio of labeled data')
    args = parser.parse_args()
    return args


def test_set_parser(publication_txt_path_prefix, publications_json_path, data_sets_json_path, output_filename):
    data_set_mention_info = []
    pub_date_dict = dict()
    logging.info("Loading data_sets.json file...")
    # open the publications.json file
    with open(data_sets_json_path) as json_data_sets:
        # parse it as JSON
        data_sets = json.load(json_data_sets)
        # loop
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
                if len(words) > 0  and len(words) <= 30:
                    formatted_mention_list.append([words, list_to_string(words)])
            data_set_mention_info.append([date, data_set_id, formatted_mention_list])
    data_set_mention_info.sort(key=lambda x: int(x[0]), reverse=True)
    # set prefix to formatted publication txt files
    formatted_txt_path_prefix = "./formatted-data/"
    # set path to publications.json
    formatted_publications = dict()
    logging.info("Tokenizing publications.json file...")
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
                pub_date = '2019-01-01'
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
        logging.info("Tagging pre-found dataset mentions in publications...")
        output = extract_formatted_data_test(formatted_publications, data_set_mention_info, pub_date_dict)
        logging.info("Writing on new csv file...")
        writer.writerows(output)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%m-%d %H:%M')
args = get_args()
test_set_parser(args.publication_txt_path_prefix, args.publications_json_path, args.data_sets_json_path, args.output_filename)
