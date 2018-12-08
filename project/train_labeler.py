from util import get_vocab, embed_indexed_sequence, \
		get_word2idx_idx2word, get_embedding_matrix, read_pub_json_files, write_predictions
from util import TextDatasetWithGloveElmoSuffix as TextDataset
from util import evaluate
from models import RNNSequenceModel
from util import list_to_string, evaluate_clf_cnn, normalize_string, split_into_sentences
import torch.nn.functional as F
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import re

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
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--data_sets_json_path', type=str, default='./formatted-data/data_sets.json',
											help='test data path')
	parser.add_argument('--glove_path', type=str, default='./glove/glove840B300d.txt',
											help='glove path (default: ./glove/glove840B300d.txt)')
	parser.add_argument('--tag_file_path', type=str, default='./formatted-data/datsetIds', 
											help='tagfile path (default: ./formatted-data/datsetIds)')
	parser.add_argument('--batch_size', type=int, default=128,
											help='batch size (default: 128)')
	parser.add_argument('--rnn_model_path', type=str, default='./checkpoint/rcc_labeler.pkl',
											help='file path for torch model states (default: ./checkpoint/rcc_labeler.pkl)')
	parser.add_argument('--lr', type=float, default=0.01)
	parser.add_argument('--num_epochs', type=int, default=1)
	parser.add_argument('--log_interval', type=int, default=100)
	parser.add_argument('--train_preprocessed', type=str, default='./formatted-data/rcc_corpus_train.csv',
											help='processed test data path')
	parser.add_argument('--train_bruteforced', type=str, default='./formatted-data/rcc_corpus_train_by_bruteforce.csv',
											help='processed test data path')
	parser.add_argument('--hidden_size', type=int, default=300)
	args = parser.parse_args()
	return args

logging.basicConfig(level=logging.DEBUG,
										format='%(asctime)s %(message)s',
										datefmt='%m-%d %H:%M')
logging.info("PyTorch version: {}".format(torch.__version__))
logging.info("GPU Detected: {}".format(torch.cuda.is_available()))
using_GPU = torch.cuda.is_available()

args = get_args()

logging.info("Loading parsed train data from {}".format(args.train_preprocessed))
raw_train_rcc = []
with open(args.train_preprocessed) as f:	# rcc_corpus_train.csv
	lines = csv.reader(f)
	next(lines)
	for line in lines:
		publication_id = int(line[0])
		word_seq = ast.literal_eval(line[1])
		label_seq = ast.literal_eval(line[2])
		label_seq = [0 if l=='_' else 2 if 'B' in l else 1 for l in label_seq] 
		assert (len(word_seq) == len(label_seq))
		labeled = 'Y'
		if len(set(label_seq)) == 1:
			labeled = 'N'
		raw_train_rcc.append([publication_id, word_seq, label_seq, labeled])

logging.info("Loading parsed train-bruteforced data from {}".format(args.train_bruteforced))
raw_train_bruteforce_rcc = []
with open(args.train_bruteforced) as f:	# rcc_corpus_train_by_bruteforce.csv
	lines = csv.reader(f)
	next(lines)
	for line in lines:
		publication_id = int(line[0])
		word_seq = ast.literal_eval(line[1])
		label_seq = ast.literal_eval(line[2])
		label_seq = [0 if l=='_' else 2 if 'B' in l else 1 for l in label_seq] 
		labeled = line[3]
		assert (len(word_seq) == len(label_seq))
		raw_train_bruteforce_rcc.append([publication_id, word_seq, label_seq, labeled])

train_rcc = []
for raw_sent, raw_sent_brute in zip(raw_train_rcc, raw_train_bruteforce_rcc):
	pub_id = raw_sent[0]
	word_seq = raw_sent[1]
	label_seq = raw_sent[2]
	if raw_sent[3] == 'N' and raw_sent_brute[3] == 'Y':
		label_seq = raw_sent_brute[2]
	assert (len(word_seq) == len(label_seq))
	train_rcc.append([word_seq, label_seq, pub_id])

vocab = get_vocab(train_rcc)
word2idx, idx2word = get_word2idx_idx2word(vocab)
logging.info("Loading glove embeddings")
glove_embeddings = get_embedding_matrix(word2idx, idx2word, args, normalization=False)

if using_GPU:
	elmo = ElmoEmbedder("./elmo/options.json", "./elmo/weights.hdf5", 0)
else:
	elmo = ElmoEmbedder("./elmo/options.json", "./elmo/weights.hdf5", -1)

# logging.info("embedd test data with glove and elmo vectors")
# embedded_rcc = []
# for example in tqdm(train_rcc, total=len(train_rcc)):
# 	embedded_rcc.append([embed_indexed_sequence(example[0], word2idx, glove_embeddings, elmo), example[1]])

# pickle.dump(embedded_rcc, open('./rnn_training_embedd.data', "wb+"), protocol=-1)
# with open('./rnn_training_embedd.data', "rb+") as infile:
#  embedded_rcc = pickle.load(infile)

import random
random.shuffle(train_rcc)
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

RNN_Seq = RNNSequenceModel(3, 300 + 1024 + 4, hidden_size=args.hidden_size, num_layers=1, bidir=True)
loss_criterion = nn.NLLLoss(reduction='elementwise_mean')
if using_GPU:
	RNN_Seq = RNN_Seq.cuda()
	state = torch.load(args.rnn_model_path)
else:
	state = torch.load(args.rnn_model_path, map_location='cpu')

if torch.cuda.device_count() > 1:
		RNN_Seq= nn.DataParallel(RNN_Seq)
		logging.info("Load RNN_Seq Model")
		RNN_Seq.load_state_dict(state['state_dict'])
RNN_Seq.to(device)

optimizer = torch.optim.SGD(RNN_Seq.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=1e-5)
#optimizer = torch.optim.Adam(RNN_Seq.parameters(), lr=args.lr) #, weight_decay=1e-5, amsgrad=True)
optimizer.load_state_dict(state['optimizer'])
for param_group in optimizer.param_groups:
	param_group['lr'] = args.lr
best_val_loss = state['loss']
#best_val_loss = 0
prev_loss = best_val_loss
logging.info("*" * 50)
logging.info(RNN_Seq)
logging.info("*" * 50)
sentences = [example[0] for example in train_rcc]
label_seqs = [example[1] for example in train_rcc]

thousand_folds = []
fold_size = int(len(train_rcc) / 1000)
for i in range(1000):
	thousand_folds.append((sentences[i * fold_size:(i + 1) * fold_size], label_seqs[i * fold_size:(i + 1) * fold_size]))

losses = []
accuracies = []
learning_rates = []
epochs = []

epoch_base = -args.num_epochs
num_iter = 0
RNN_Seq.train()
logging.info("Thousand Fold Validation, 1 fold size: {}".format(fold_size))
logging.info("Training with Batch Size: {}, Learning Rate: {}".format(args.batch_size, args.lr))
logging.info("Loaded best validation loss: {}".format(best_val_loss))
logging.info("*" * 50)
prev_train_loss = 999999

try:
	while True:
		for i in range(1000):
			logging.info("1000 fold validation turn change...")
			first_in_fold = True
			training_sentences = []
			training_labels = []
			for j in range(1000):
				if j != i:
					training_sentences.extend(thousand_folds[j][0])
					training_labels.extend(thousand_folds[j][1])
			training_dataset_rcc = TextDataset(training_sentences, training_labels, word2idx, glove_embeddings, elmo)
			val_dataset_rcc = TextDataset(thousand_folds[i][0], thousand_folds[i][1], word2idx, glove_embeddings, elmo)
			train_dataloader_rcc = DataLoader(dataset=training_dataset_rcc, batch_size=args.batch_size, shuffle=True, collate_fn=TextDataset.collate_fn)
			val_dataloader_rcc = DataLoader(dataset=val_dataset_rcc, batch_size=args.batch_size, shuffle=False, collate_fn=TextDataset.collate_fn)
			epoch_base += args.num_epochs
			for epoch in range(args.num_epochs):
				logging.info("Starting epoch {}".format(epoch + 1 + epoch_base))
				for (example_text, example_lengths, labels) in train_dataloader_rcc:
					if num_iter % 20 == 0:
						logging.info("Iteration {}".format(num_iter))
					example_text = Variable(example_text)
					example_lengths = Variable(example_lengths)
					labels = Variable(labels)
					if using_GPU:
							example_text = example_text.cuda()
							example_lengths = example_lengths.cuda()
							labels = labels.cuda()
					# predicted shape: (batch_size, seq_len, 2)
					predicted = RNN_Seq(example_text, example_lengths)
					batch_loss = loss_criterion(predicted.view(-1, 3), labels.view(-1))
					optimizer.zero_grad()
					batch_loss.backward()
					optimizer.step()
					num_iter += 1
					if num_iter % args.log_interval == 0:
						avg_eval_loss, accuracy = evaluate(val_dataloader_rcc, RNN_Seq, loss_criterion, using_GPU)
						print("Iteration {}. Validation Loss: {}, Accuracy: {:.4f}".format(num_iter, avg_eval_loss, accuracy))
						if best_val_loss > avg_eval_loss:
							logging.info("New Best Validation Loss! {}, before: {}".format(avg_eval_loss, best_val_loss))
							best_val_loss = avg_eval_loss
							logging.info("Save Checkpoint to {}".format(args.rnn_model_path))
							torch.save({
													'state_dict': RNN_Seq.state_dict(),
													'optimizer': optimizer.state_dict(),
													'loss': best_val_loss,
													}, args.rnn_model_path)
						elif best_val_loss == 0:
							logging.info("Initial Validation Loss! {}".format(avg_eval_loss))
							best_val_loss = avg_eval_loss
							logging.info("Save Checkpoint to {}".format(args.rnn_model_path))
							torch.save({
													'state_dict': RNN_Seq.state_dict(),
													'optimizer': optimizer.state_dict(),
													'loss': best_val_loss,
													}, args.rnn_model_path)
						# elif not first_in_fold and avg_eval_loss >= prev_loss:
						# 	for param_group in optimizer.param_groups:
						# 		prev_lr = param_group['lr']
						# 		param_group['lr'] = prev_lr/10
						# 		logging.info('gradient decay from {} to {}'.format(prev_lr, param_group['lr']))
						losses.append(avg_eval_loss)
						accuracies.append(accuracy)
						temp_lr = 0
						for param_group in optimizer.param_groups:
							temp_lr = param_group['lr']
						learning_rates.append(temp_lr)
						epochs.append(epoch)
						prev_loss = avg_eval_loss
						first_in_fold = False
except KeyboardInterrupt:
	logging.info('interrupted, dumping file to ./l_a_lr_ep_rnn.data')
	pickle.dump([losses, accuracies, learning_rates, epochs], open('./l_a_lr_ep_rnn.data', "wb+"), protocol=-1)

























