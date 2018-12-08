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
    parser.add_argument('--kernel_sizes', type=str, default='2,3,4,5',
                        help='comma-separated kernel size to use for convolution (default: 2,3,4,5)')
    parser.add_argument('--kernel_num', type=int, default=100,
                        help='number of each kind of kernel (default: 100)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size (default: 128)')
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

RNNseq_model = RNNSequenceModel(num_classes=3, embedding_dim=300 + 1024 + 4, hidden_size=300, num_layers=1, bidir=True)
if using_GPU:
  RNNseq_model = RNNseq_model.cuda()
  state_dict = torch.load(args.rnn_model_path)['state_dict']
else:
  state_dict = torch.load(args.rnn_model_path, map_location='cpu')['state_dict']
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
RNNseq_model.load_state_dict(new_state_dict)
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
class_to_idx = {}
with open(args.tag_file_path, 'r') as f:
  for (n, i) in enumerate(f): #idx, label
    class_to_idx[i.strip()] = n
CNN_Text = CNN_Text(args.kernel_num, args.kernel_sizes, len(class_to_idx), 300 + 1024 + 4)
loss_criterion = nn.NLLLoss()
if using_GPU:
  CNN_Text = CNN_Text.cuda()
  state_dict = torch.load(args.cnn_model_path)['state_dict']
else:
  state_dict = torch.load(args.cnn_model_path, map_location='cpu')['state_dict']
CNN_Text.load_state_dict(state_dict)