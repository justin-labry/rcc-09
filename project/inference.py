from util import get_vocab, embed_indexed_sequence, \
    get_word2idx_idx2word, get_embedding_matrix, read_pub_json_files, write_predictions
from util import TextDatasetWithGloveElmoSuffix_ForTest as RNN_Testset
from util import TextDatasetForClassfier_CNN_ForTest as CNN_Testset
from util import evaluate
from models import RNNSequenceModel, CNN_Text
from util import list_to_string
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import argparse
import logging
import pickle
import csv
import ast
import json
import h5py
from tqdm import tqdm
from allennlp.commands.elmo import ElmoEmbedder
from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser(description='rcc-09')
    parser.add_argument('--pub_info_path', type=str, default='../data/input/publications.json',
                        help='publication.json path for train data (default: ../data/input/publications.json)')
    parser.add_argument('--rnn_model_path', type=str, default='./checkpoint/rcc_labeler.pkl',
                        help='file path for torch model states (default: ./checkpoint/rcc_labeler.pkl)')
    parser.add_argument('--cnn_model_path', type=str, default='./checkpoint/rcc_classifier_cnn.pkl')
    parser.add_argument('--test_preprocessed', type=str, default='./formatted-data/rcc_corpus_test.csv',
                        help='processed test data path')
    parser.add_argument('--not_found_test_path', type=str, default='./formatted-data/rcc_test.csv',
                        help='processed test data path, extracted only the unlabeled lines')
    parser.add_argument('--tag_file_path', type=str, default='./formatted-data/datsetIds', 
    										help='tagfile path (default: ./formatted-data/datsetIds)')
    parser.add_argument('--kernel_sizes', type=str, default='1,2,3,4,5',
                        help='comma-separated kernel size to use for convolution (default: 2,3,4,5)')
    parser.add_argument('--kernel_num', type=int, default=800,
                        help='number of each kind of kernel (default: 100)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size (default: 1)')
    parser.add_argument('--unordered_output_path', type=str, default='./formatted-data/model_output_mentions.csv')
    parser.add_argument('--vocab_info_path', type=str, default='./formatted-data/vocabInfo.data', 
                        help='load word2idx, idx2word from dump')
    parser.add_argument('--data_sets_json_path', type=str, default='./formatted-data/data_sets.json')
    parser.add_argument('--glove_path', type=str, default='./glove/glove840B300d.txt',
                        help='glove path (default: ./glove/glove840B300d.txt)')
    args = parser.parse_args()
    return args

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')

logging.info("PyTorch version: {}".format(torch.__version__))
logging.info("GPU Detected: {}".format(torch.cuda.is_available()))
using_GPU = torch.cuda.is_available()

args = get_args()

logging.info("Loading parsed test data from {}".format(args.test_preprocessed))
raw_test_rcc = []
test_annotated = []
with open(args.test_preprocessed) as f:
	lines = csv.reader(f)
	next(lines)
	for line in lines:
		publication_id = int(line[0])
		word_seq = ast.literal_eval(line[1])
		label_seq = ast.literal_eval(line[2])
		label_seq = [0 if l=='_' else 2 if 'B' in l else 1 for l in label_seq] 
		labeled = line[3]
		assert (len(word_seq) == len(label_seq))
		if labeled == 'N':
			raw_test_rcc.append([word_seq, publication_id])
		else:
			test_annotated.append([word_seq, label_seq, publication_id])

logging.info("Write other sentences to {}".format(args.not_found_test_path))
with open(args.not_found_test_path, 'w') as csvfile:
  fieldnames = [
      'publication_id',
      'sentence']
  writer = csv.DictWriter(
      csvfile,
      fieldnames=fieldnames,
      quoting=csv.QUOTE_ALL)
  writer.writeheader()
  output = []
  for word_seq, publication_id in raw_test_rcc:
  	output.append({'publication_id': publication_id,
  								 'sentence': word_seq})
  logging.info("Writing on new csv file...")
  writer.writerows(output)

logging.info('size of test set: {}, annotated by brute-force test set: {}, to-be-found test set: {}'.format(
                len(raw_test_rcc) + len(test_annotated), len(test_annotated), len(raw_test_rcc)))

# logging.info("Read vocabulary info from {}".format(args.vocab_info_path))
# with open(args.vocab_info_path, "rb+") as infile:
#   word2idx, idx2word = pickle.load(infile)

vocab = get_vocab(raw_test_rcc + test_annotated)
word2idx, idx2word = get_word2idx_idx2word(vocab)
logging.info("Loading glove embeddings")
glove_embeddings = get_embedding_matrix(word2idx, idx2word, args, normalization=False)

if using_GPU:
  elmo = ElmoEmbedder("./elmo/options.json", "./elmo/weights.hdf5", 0)
else:
  elmo = ElmoEmbedder("./elmo/options.json", "./elmo/weights.hdf5", -1)

############
# labeling #
############
embedded_test_rcc = []
logging.info("embedd test data with glove and elmo vectors")
for example in tqdm(raw_test_rcc, total=len(raw_test_rcc)):
	embedded_test_rcc.append([example[1], embed_indexed_sequence(example[0], word2idx, glove_embeddings, elmo)])

# pickle.dump(embedded_test_rcc, open('./labeler_embedd_temp.data', "wb+"), protocol=-1)
# with open('./labeler_embedd_temp.data', "rb+") as infile:
#  embedded_test_rcc = pickle.load(infile)

logging.info("Set up Dataloader")
test_dataset_rcc = RNN_Testset([example[0] for example in embedded_test_rcc], # pub_id
															 [example[1] for example in embedded_test_rcc]) # embedded sentence 
# Set up a DataLoader for the test dataset
test_dataloader_rcc = DataLoader(dataset=test_dataset_rcc, batch_size=args.batch_size,
                                 collate_fn=RNN_Testset.collate_fn)

logging.info("*" * 25 + " Mention Labeling By LSTM tagging model " + "*" * 25)

RNNseq_model = RNNSequenceModel(num_classes=3, embedding_dim=300 + 1024 + 4, hidden_size=300, num_layers=1, bidir=True)
if using_GPU:
  RNNseq_model = RNNseq_model.cuda()
  state_dict = torch.load(args.rnn_model_path)['state_dict']
else:
  state_dict = torch.load(args.rnn_model_path, map_location='cpu')['state_dict']
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
RNNseq_model.load_state_dict(new_state_dict)
result = write_predictions(raw_test_rcc, test_dataloader_rcc, RNNseq_model, using_GPU, args.not_found_test_path)
logging.info("Write predictions to {}".format(args.not_found_test_path))
f = open(args.not_found_test_path, 'w')
writer = csv.writer(f)
writer.writerows(result)
f.close()
logging.info("*" * 25 + " Mention Labeling By LSTM tagging model " + "*" * 25)


##############
# classfying #
##############
logging.info("*" * 25 + " Dataset Recognition By CNN Text Classifier " + "*" * 25)
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
logging.info("Loading tagFile")
idx_to_class = {}
class_to_idx = {}
with open(args.tag_file_path, 'r') as f:
  for (n, i) in enumerate(f): #idx, label
    class_to_idx[i.strip()] = n
    idx_to_class[n] = i.strip()
dataset_id_to_date = {}
logging.info("Loading data_sets.json file for dataset info")
with open(args.data_sets_json_path) as json_data_sets:
    data_sets = json.load(json_data_sets)
    for data_set_info in data_sets:
        data_set_id = data_set_info.get( "data_set_id", None )
        date = data_set_info.get( "date", None )
        date.encode('ascii','ignore')
        if 'None' in date:
            date = '1800-01-01'
        date = date[:4]
        dataset_id_to_date[data_set_id] = date
CNN_Text = CNN_Text(args.kernel_num, args.kernel_sizes, len(class_to_idx), 300 + 1024 + 4)
loss_criterion = nn.NLLLoss()
if using_GPU:
  CNN_Text = CNN_Text.cuda()
  state_dict = torch.load(args.cnn_model_path)['state_dict']
else:
  state_dict = torch.load(args.cnn_model_path, map_location='cpu')['state_dict']
# if torch.cuda.device_count() > 1:
#     CNN_Text= nn.DataParallel(CNN_Text)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
CNN_Text.load_state_dict(new_state_dict)
pub_date_dict = {}
with open(args.pub_info_path) as json_publication_file:
  publication_list = json.load(json_publication_file)
  for publication_info in publication_list:
    publication_id = publication_info.get( "publication_id", None )
    unique_identifier = publication_info.get( "unique_identifier", None ) # id가 bbk로 시작하면 pub_date은 None임
    if 'bbk' not in unique_identifier:
      pub_date = publication_info.get( "pub_date", None )
    else:
      pub_date = '2019-00-00'
    pub_date.encode('ascii', 'ignore')
    pub_date = int(pub_date[:4])
    pub_date_dict[publication_id] = pub_date
found_mentions = []
with open(args.not_found_test_path) as f:
  lines = csv.reader(f)
  next(lines)
  for line in lines:
    publication_id = int(line[0])
    word_seq = ast.literal_eval(line[1])
    label_seq = ast.literal_eval(line[2])
    assert (len(word_seq) == len(label_seq))
    found = False
    mentions_words = []
    for w, l in zip(word_seq, label_seq):
      if l == 2:
        mentions_words.append(w)
        found = True
      elif l == 1 and found:
        mentions_words.append(w)
      else:
        if found:
          found_mentions.append([mentions_words, publication_id, pub_date_dict[publication_id]])
          mentions_words = []
          found = False
    if found:
      found_mentions.append([mentions_words, publication_id, pub_date_dict[publication_id]])
logging.info("embedd test data with glove and elmo vectors")
embedded_test_rcc = []
for example in tqdm(found_mentions, total=len(found_mentions)):
  embedded_test_rcc.append([embed_indexed_sequence(example[0], word2idx, glove_embeddings, elmo), example[1], example[2], example[0]])

#pickle.dump(embedded_test_rcc, open('./clf_embedd_temp.data', "wb+"), protocol=-1)

test_dataset_rcc = CNN_Testset([example[0] for example in embedded_test_rcc], # embedded sentence
                               [example[1] for example in embedded_test_rcc], # pub_id
                               [example[2] for example in embedded_test_rcc], # pub_date
                               [example[3] for example in embedded_test_rcc]  # word_seq
                               )
test_dataloader_rcc = DataLoader(dataset=test_dataset_rcc, batch_size=args.batch_size,
                                 collate_fn=CNN_Testset.collate_fn)
output = []
CNN_Text.eval()
max_score = 0
logging.info("Classify datasets from captured mentions...")
for example_text, pub_ids, pub_dates, word_seqs in tqdm(test_dataloader_rcc):
  example_text = Variable(example_text)
  if using_GPU:
    example_text = example_text.cuda()
  with torch.no_grad():
    logit = CNN_Text(example_text)
    logit = F.softmax(logit, dim=1)
  max_probs10, predicted_labels10 = torch.topk(logit, 10, dim=1)
  for i in range(len(pub_ids)):
    predictions = predicted_labels10[i]
    prediction = predictions[-1]
    pub_date = pub_dates[i]
    score = 0
    for j in range(10):
      if predictions[j] not in dataset_id_to_date:
        dataset_id_to_date[predictions[j]] = '1800'
        prediction = predictions[j]
        score += max_probs10[i][j].item()
        break
      elif pub_date >= int(dataset_id_to_date[predictions[j]]):
        prediction = predictions[j]
        score += max_probs10[i][j].item()
        break
      else:
        score += max_probs10[i][j].item()
    dataset_id = idx_to_class[prediction.item()]
    if score > max_score:
      max_score = score
    output.append([word_seqs[i], pub_ids[i], dataset_id, score])
output_normalized = []
for example in output:
  normalizer = 1/float(max_score)
  output_normalized.append([example[0], example[1], example[2], normalizer * example[3]])

with open(args.unordered_output_path, 'w') as csvfile:
  fieldnames = [
      'mention',
      'publication_id',
      'dataset_id',
      'score']
  writer = csv.DictWriter(
      csvfile,
      fieldnames=fieldnames,
      quoting=csv.QUOTE_ALL)
  writer.writeheader()
  output = []
  for m, p, d, s in output_normalized:
    output.append({'mention': m,
                   'publication_id': p,
                   'dataset_id': d,
                   'score': s})
  logging.info("Writing on new csv file...")
  writer.writerows(output)
logging.info("*" * 25 + " Dataset Recognition By CNN Text Classifier " + "*" * 25)
