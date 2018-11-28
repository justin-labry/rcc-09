# imports
import codecs
import json
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

def normalize_string(text):
    text = re.sub("[^ ]+%", ' ', text)
    text = re.sub('\\\\\\\"', ' ', text)
    text = re.sub(r"[^a-zA-Z0-9()/;:#,.?!&=~\-@\"\' ]+", ' ', text)
    text = re.sub("\s\s+", " ", text)
    # remove hyphens
    text = re.sub('\xad', '-', text)
    text = re.sub('\u00ad', '-', text)
    text = re.sub('\N{SOFT HYPHEN}', '-', text)
    text = re.sub(r'([^\s])- ', "\\1", text)
    text = re.sub(r'\-', ' ', text)
    text = re.sub("\s\s+", " ", text)
    # remove quotes
    text = re.sub('\"', ' ', text)
    text = re.sub("\s\s+", " ", text)
    # etc
    text = re.sub(r"(?!\([1-2][0-9][0-9][0-9] to [1-2][0-9][0-9][0-9]\))\([^\)a-zA-Z]+to[^\)a-zA-Z]+\)", ' ', text) #(241234 to 124323)
    text = re.sub(r"\([0-9]+\.[0-9]+ in [0-9]+\.[0-9]+\)", ' ', text) #(4.3-1.2)
    text = re.sub(r"(?!\([1-2][0-9][0-9][0-9]\))(?!\([1-2][0-9][0-9][0-9] [^\)]+\))\([0-9\.\,\+ ]+\)", ' ', text)
    text = re.sub(r"(?! [1-2][0-9][0-9][0-9] ) [0-9][0-9\.\,]+[0-9] ", ' ', text)
    text = re.sub(r" [0-9]+\.[0-9]+", ' ', text)
    text = re.sub("\s\s+", " ", text)
    return text



def label_substring(sentence, mention, label_sequence, data_set_id):
    found_cnt = 0
    for i, word in enumerate(sentence[:-len(mention) + 1]):
        if all(x == y for x, y in zip(mention, sentence[i:i + len(mention)])):
            if label_sequence[i] is '_':
                found_cnt += 1
                label_sequence[i] = 'B-' + str(data_set_id)                
            for j in range(len(mention)-1):
                label_sequence[i + j + 1] = 'I'
    return label_sequence, found_cnt


def encode(sentences, mention_list):
    label_sequence_list = []
    found_cnt = 0
    for sentence in sentences:
        label_sequence = ['_' for _ in sentence]
        for data_set_id, mentions in mention_list:
            for mention in mentions:
                label_sequence, cnt = label_substring(sentence, mention, label_sequence, data_set_id)
                found_cnt += cnt
        label_sequence_list.append(label_sequence)
    return label_sequence_list, found_cnt


def encode_test(raw_text, data_set_mention_info, pub_date):
    label_sequence = ['_' for _ in range(len(raw_text.split()))]
    for date, data_set_id, mention_list in data_set_mention_info:   # date이 큰 것부터(최신부터) 들어오도록 정렬되어 있음
        if date < pub_date:
            for mention_info in mention_list:
                mention, raw_mention = mention_info[0], mention_info[1]
                substitute_pattern = '<MT-B>' + ' <MT-I>' * (len(mention)-1)
                replaced_text = raw_text.replace(raw_mention, substitute_pattern)
                begin_mark = 'B-' + str(data_set_id)
                res = [begin_mark if w=='<MT-B>' else 'I' if w=='<MT-I>' else '_' for w in replaced_text.split()]
                label_sequence = [prev_l if curr_l=='_' else curr_l if prev_l=='_' else 'I' if prev_l=='I' or curr_l=='I' else prev_l for prev_l, curr_l in zip(label_sequence, res)]
    return label_sequence


def extract_formatted_data(formatted_publications, citation_dict):
    output = []
    found_cnt = 0
    for publication_id, sentences in tqdm(formatted_publications.items(), total=len(formatted_publications)):
        if publication_id in citation_dict:
            mention_list = citation_dict[publication_id] # [[data_set_id, formatted_mention_list]]
            label_sequence_list, cnt = encode(sentences, mention_list)
            found_cnt += cnt
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
    print("found mentions: ", end='')
    print(found_cnt)
    return output


def extract_formatted_data_test(formatted_publications, data_set_mention_info, pub_date_dict):
    output = []
    found_cnt = 0
    for publication_id, [sentences, raw_text] in tqdm(formatted_publications.items(), total=len(formatted_publications)):
        label_sequence = encode_test(raw_text, data_set_mention_info, pub_date_dict[publication_id])
        startidx = 0
        for sentence in sentences:
            sub_label_sequence = label_sequence[startidx:startidx+len(sentence)]
            startidx += len(sub_label_sequence)
            if sub_label_sequence[0] == 'I':
                for i, l in enumerate(sub_label_sequence):
                    if l == 'I':
                        sub_label_sequence[i] = '_'
                    else:
                        break
            labeled = 'N'
            if len(set(sub_label_sequence)) != 1:
                labeled = 'Y'
            output.append({'publication_id': publication_id,
                           'sentence': sentence,
                           'label_sequence': sub_label_sequence,
                           'labeled': labeled})
        for l in label_sequence:
            if 'B' in l:
                found_cnt += 1
    print("found mentions: ", end='')
    print(found_cnt)
    return output


def train_set_parser():
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
                mention = normalize_string(mention)
                sentences = split_into_sentences(mention)
                words = []
                for sentence in sentences:
                    words += word_tokenize(sentence)
                words = [w for w in words if len(w)<20]
                if len(words) > 0:
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
                words = [w for w in words if len(w)<20]
                if len(words) >= 10 and len(words) <= 50:
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


def test_set_parser(data_set_path, output_filename):
    # set path to data_set_citations.json
    data_sets_json_path = "../data/input/data_sets.json"
    data_set_mention_info = []
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
                words = [w for w in words if len(w)<20]
                if len(words) > 0:
                    formatted_mention_list.append([words, list_to_string(words)])
            data_set_mention_info.append([date, data_set_id, formatted_mention_list])
    data_set_mention_info.sort(key=lambda x: int(x[0]), reverse=True)
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
            pub_date = int(pub_date[:4]) * 12 * 31 + int(pub_date[5:7]) * 31 + int(pub_date[8:10])
            # get raw text
            raw_text = ''
            txt_file_path = data_set_path + 'text/' + text_file_name
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
                words = [w for w in words if len(w)<20]
                if len(words) >= 10 and len(words) <= 50:
                    formatted_text_list.append(words)
                    chopped_raw_text += ' ' + list_to_string(words)
            formatted_publications[publication_id] = [formatted_text_list, chopped_raw_text.strip()] 
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
        output = extract_formatted_data_test(formatted_publications, data_set_mention_info, pub_date_dict)
        print("Writing on new csv file...", end='')
        writer.writerows(output)
        print("DONE")


# for trainset
train_set_parser()
# for devset
publication_txt_path_prefix = './dev_data/'
output_filename = 'rcc_corpus_dev.csv'
test_set_parser(publication_txt_path_prefix, output_filename)

