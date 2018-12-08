#import split as sp
import csv
import ast
from tqdm import tqdm


FORMATTED_DATA_PATH = './formatted-data/'
ABSTRACT_DATA_PATH = './formatted-data/abstracts/'

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

with open(FORMATTED_DATA_PATH + 'rcc_corpus_test.csv') as f:
    lines = csv.reader(f)
    next(lines)
    
    write_dict =dict()
    write_count = dict()
    for line in lines:
        word_id = int(line[0])        # list of tokens(words)
        write_dict[word_id] = ''
        write_count[word_id] = 0
    f.close()

with open(FORMATTED_DATA_PATH + 'rcc_corpus_test.csv') as f1:
    lines1 = csv.reader(f1)
    next(lines1)

    for line in lines1:
        word_id = int(line[0])
        list_ = ast.literal_eval(line[1])
        string = list_to_string(list_)
        if write_count[word_id] == 40:
            continue
        else:
            write_dict[word_id] = write_dict[word_id] + ' ' + string
            write_count[word_id] = write_count[word_id] + 1
    f1.close()

for key, value in write_dict.items():
    f2 = open(ABSTRACT_DATA_PATH + str(key) + '.txt', 'w')
    f2.write(value)
