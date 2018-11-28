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


# set path to data_set_citations.json
data_sets_json_path = "../data/input/data_sets.json"
all_mention_list = []
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
        mention_list = data_set_info.get( "mention_list", None )
        name = re.sub('\\\\\\\"', '\"', name)
        name_words = word_tokenize(name)
        all_mention_list.append(name)
        for mention in mention_list:
            mention.encode('ascii','ignore')
            mention = re.sub('\\\\\\\"', '\"', mention)
            if all(c.islower() for c in mention) and len(words) <= 2:
                continue    # to avoid pronoun mentions like 'data', 'time'
            sentences = split_into_sentences(mention)
            words = []
            for sentence in sentences:
                words += word_tokenize(sentence)
            all_mention_list.append(sentence)

cnt = 0
for mention in all_mention_list:
    new = normalize_string(mention).strip()
    if len(mention.split()) != len(new.split()):
        print(mention)
        print(normalize_string(mention))
        cnt += 1
print(cnt)



