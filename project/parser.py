# imports
import codecs
import json
#from nltk.parse.corenlp import CoreNLPParser
from nltk.tokenize import word_tokenize
import re
import csv
import unicodedata
from tqdm import tqdm


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


def split_into_sentences(text):
    """
    This function can split the entire text of Huckleberry Finn into sentences in about 0.1 seconds
    and handles many of the more painful edge cases that make sentence parsing non-trivial
    """
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"
    digits = "([0-9]+)"
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub("\s\s+", " ", text)
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ", text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" "+suffixes+"[.] "+starters, " \\1<stop> \\2", text)
    text = re.sub(" "+suffixes+"[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    text = re.sub("\s\s+", " ", text)
    sentences = text.split("<stop>")
    #sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def unify_hyphens(text):
    text = text.replace('\xad', '-')
    text = text.replace('\u00ad', '-')
    text = text.replace('\N{SOFT HYPHEN}', '-')
    text = re.sub('([^\s])- ', "\\1-",text)
    return text


def label_substring(sentence, mention, label_sequence, data_set_id):
    for i, word in enumerate(sentence[:-len(mention) + 1]):
        if all(x == y for x, y in zip(mention, sentence[i:i + len(mention)])):
            if label_sequence[i] is '_':
                label_sequence[i] = 'B-' + str(data_set_id)                
            for j in range(len(mention)-1):
                label_sequence[i + j + 1] = 'I'
    return label_sequence


def label_substring_test(sentence, mention, label_sequence, labeled, data_set_id, pub_date, data_set_date_dict):
    for i, word in enumerate(sentence[:-len(mention) + 1]):
        if all(x == y for x, y in zip(mention, sentence[i:i + len(mention)])):
            curr_data_set_date = data_set_date_dict[data_set_id]
            if label_sequence[i] is '_':
                label_sequence[i] = 'B-' + str(data_set_id)
            elif 'I' in label_sequence[i]:
                continue
            else: # B-id
                prev_data_set_date = data_set_date_dict[int(label_sequence[i][2:])]
                if prev_data_set_date > curr_data_set_date:
                    continue
                else:
                    label_sequence[i] = 'B-' + str(data_set_id)
            for j in range(len(mention)-1):
                label_sequence[i + j + 1] = 'I-' + str(data_set_id)
            labeled = 'Y'
            print(label_sequence)
    return label_sequence, labeled


# [[data_set_id, formatted_mention_list]]
# [sentences]
def encode(sentences, mention_list):
    label_sequence_list = []
    for sentence in sentences:
        label_sequence = ['_' for _ in sentence]
        for data_set_id, mentions in mention_list:
            for mention in mentions:
                label_sequence = label_substring(sentence, mention, label_sequence, data_set_id)
        label_sequence_list.append(label_sequence)
    return label_sequence_list


def encode_test(sentences, data_set_mention_dict, pub_date, data_set_date_dict):
    label_sequence_list = []
    labeled_list = []
    for sentence in sentences:
        label_sequence = ['_' for _ in sentence]
        labeled = 'N'
        for data_set_id, mention_list in data_set_mention_dict.items():
            if pub_date < data_set_date_dict[data_set_id]:
                continue
            for mention in mention_list:
                label_sequence, labeled = label_substring_test(sentence, mention, label_sequence, labeled, data_set_id, pub_date, data_set_date_dict)
        label_sequence = ['I' if 'I' in l else l for l in label_sequence]
        label_sequence_list.append(label_sequence)
        labeled_list.append(labeled)
    return label_sequence_list, labeled_list


def extract_formatted_data(formatted_publications, citation_dict):
    output = []
    for publication_id, sentences in tqdm(formatted_publications.items(), total=len(formatted_publications)):
        if publication_id in citation_dict:
            mention_list = citation_dict[publication_id] # [[data_set_id, formatted_mention_list]]
            label_sequence_list = encode(sentences, mention_list)
            for sentence, label_sequence in zip(sentences, label_sequence_list):
                output.append({'publication_id': publication_id,
                               'sentence': sentence,
                               'label_sequence': label_sequence,
                               'labeled': 'Y'})
        else:
            for sentence in sentences:
                label_sequence = ['_' for _ in sentence]
                output.append({'publication_id': publication_id,
                               'sentence': sentence,
                               'label_sequence': label_sequence,
                               'labeled': 'Y'})
    return output


def extract_formatted_data_test(formatted_publications, data_set_mention_dict, pub_date_dict, data_set_date_dict):
    # data_set_mention_dict[data_set_id] = [data_set_id, formatted_mention_list]
    output = []
    for publication_id, sentences in tqdm(formatted_publications.items(), total=len(formatted_publications)):
        label_sequence_list, labeled_list = encode_test(sentences, data_set_mention_dict, pub_date_dict[publication_id], data_set_date_dict)
        for sentence, label_sequence, labeled in zip(sentences, label_sequence_list, labeled_list):
            output.append({'publication_id': publication_id,
                            'sentence': sentence,
                            'label_sequence': label_sequence,
                            'labeled': labeled})
    return output


def train_set_parser():
    vocab = set()
    # set path to data_set_citations.json
    data_set_citations_json_path = "../data/input/data_set_citations.json"
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
                mention = unify_hyphens(mention)
                mention = re.sub('\\\\\\\"', '\"', mention)
                sentences = split_into_sentences(mention)
                words = []
                for sentence in sentences:
                    words += word_tokenize(sentence)
                    vocab.update(words)
                formatted_mention_list.append(words)
            if publication_id in citation_dict:
                citation_dict[publication_id].append([data_set_id, formatted_mention_list])
            else:
                citation_dict[publication_id] = [[data_set_id, formatted_mention_list]]
    # set prefix to publication txt files
    publication_txt_path_prefix = "../data/input/files/text/"
    # set prefix to formatted publication txt files
    formatted_txt_path_prefix = "./formatted-data/"
    # set path to publications.json
    publications_json_path = "../data/input/publications.json"
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
                pub_date = 'NA'
            pub_date.encode('ascii','ignore')
            # get raw text
            raw_text = ''
            txt_file_path = publication_txt_path_prefix + text_file_name
            with open(txt_file_path) as txt_file:
                for line in txt_file:
                    stripped_line = line.strip()
                    raw_text += ' ' + stripped_line
            raw_text.encode('ascii','ignore')
            raw_text = unify_hyphens(raw_text)
            # add to formatted_publications dictionary
            formatted_text_list = []
            sentences = split_into_sentences(raw_text)
            for sentence in sentences:
                words = word_tokenize(sentence)
                vocab.update(words)
                formatted_text_list.append(words)
            formatted_publications[publication_id] = formatted_text_list 
    # tag mentions in publication text and write in csv file
    with open('./formatted-data/rcc_corpus_train.csv', 'w') as csvfile:
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
        print("Tagging dataset mentions in publications...")
        output = extract_formatted_data(formatted_publications, citation_dict)
        print("Writing on new csv file...", end='')
        writer.writerows(output)
        print("DONE")
    # write collected vocabulary to a file, line by line
    with open("./formatted-data/rcc_vocab.txt", 'w') as vocabfile:
        vocab = list(vocab)
        for w in vocab:
            vocabfile.write("%s\n" % w)


def test_set_parser(data_set_path, output_filename):
    # set path to data_set_citations.json
    data_sets_json_path = "../data/input/data_sets.json"
    data_set_mention_dict = dict()
    data_set_date_dict = dict()
    pub_date_dict = dict()
    print("Loading data_sets.json file...")
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
            date = unify_hyphens(date)
            if 'None' in date:
                date = '1800-01-01'
            date = int(date[:4]) * 12 * 31 + int(date[5:7]) * 31 + int(date[8:10])
            mention_list = data_set_info.get( "mention_list", None )
            formatted_mention_list = []
            name = unify_hyphens(name)
            name = re.sub('\\\\\\\"', '\"', name)
            name_words = word_tokenize(name)
            formatted_mention_list.append(name_words)
            for mention in mention_list:
                mention.encode('ascii','ignore')
                mention = unify_hyphens(mention)
                mention = re.sub('\\\\\\\"', '\"', mention)
                sentences = split_into_sentences(mention)
                words = []
                for sentence in sentences:
                    words += word_tokenize(sentence)
                formatted_mention_list.append(words)
            data_set_date_dict[data_set_id] = date
            data_set_mention_dict[data_set_id] = formatted_mention_list
    # set prefix to formatted publication txt files
    formatted_txt_path_prefix = "./formatted-data/"
    # set path to publications.json
    publications_json_path = data_set_path + "publications.json"
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
            pub_date = unify_hyphens(pub_date)
            pub_date = int(pub_date[:4]) * 12 * 31 + int(pub_date[5:7]) * 31 + int(pub_date[8:10])
            # get raw text
            raw_text = ''
            txt_file_path = data_set_path + 'text/' + text_file_name
            with open(txt_file_path) as txt_file:
                for line in txt_file:
                    stripped_line = line.strip()
                    raw_text += ' ' + stripped_line
            raw_text.encode('ascii','ignore')
            raw_text = unify_hyphens(raw_text)
            # add to formatted_publications dictionary
            formatted_text_list = []
            sentences = split_into_sentences(raw_text)
            for sentence in sentences:
                words = word_tokenize(sentence)
                formatted_text_list.append(words)
            formatted_publications[publication_id] = formatted_text_list 
            pub_date_dict[publication_id] = pub_date
    # tag mentions in publication text and write in csv file
    output_filepath = './formatted-data/' + output_filename
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
        output = extract_formatted_data_test(formatted_publications, data_set_mention_dict, pub_date_dict, data_set_date_dict)
        print("Writing on new csv file...", end='')
        writer.writerows(output)
        print("DONE")


# for trainset
# train_set_parser()
# for devset
publication_txt_path_prefix = './dev_data/'
output_filename = 'rcc_corpus_dev.csv'
test_set_parser(publication_txt_path_prefix, output_filename)

