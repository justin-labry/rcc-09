from util import get_vocab, embed_indexed_sequence, \
		get_word2idx_idx2word, get_embedding_matrix, read_pub_json_files, write_predictions
from util import TextDatasetForClassfier_CNN as TextDataset
from util import evaluate
from models import CNN_Text
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
import signal
import sys
import time


def get_args():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--data_sets_json_path', type=str, default='./formatted-data/data_sets.json',
											help='test data path')
	parser.add_argument('--glove_path', type=str, default='./glove/glove840B300d.txt',
											help='glove path (default: ./glove/glove840B300d.txt)')
	parser.add_argument('--tag_file_path', type=str, default='./formatted-data/datsetIds', 
											help='tagfile path (default: ./formatted-data/datsetIds)')
	parser.add_argument('--kernel_sizes', type=str, default='1,2,3,4,5',
											help='comma-separated kernel size to use for convolution (default: 2,3,4,5)')
	parser.add_argument('--kernel_num', type=int, default=800,
											help='number of each kind of kernel (default: 100)')
	parser.add_argument('--batch_size', type=int, default=1024,
											help='batch size (default: 128)')
	parser.add_argument('--cnn_model_path', type=str, default='./checkpoint/rcc_classifier_cnn.pkl')
	parser.add_argument('--lr', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=13)
	parser.add_argument('--log_interval', type=int, default=100)
	args = parser.parse_args()
	return args

logging.basicConfig(level=logging.DEBUG,
										format='%(asctime)s %(message)s',
										datefmt='%m-%d %H:%M')
logging.info("PyTorch version: {}".format(torch.__version__))
logging.info("GPU Detected: {}".format(torch.cuda.is_available()))
using_GPU = torch.cuda.is_available()
args = get_args()
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
# mention_info = []
# logging.info("Loading mentions from {}".format(args.data_sets_json_path))
# with open(args.data_sets_json_path) as json_data_sets:
# 	# parse it as JSON
# 	data_sets = json.load(json_data_sets)
# 	# loop
# 	for data_set_info in tqdm(data_sets, total=len(data_sets)):
# 		data_set_id = data_set_info.get( "data_set_id", None )
# 		name = data_set_info.get( "name", None )
# 		name.encode('ascii','ignore')
# 		mention_list = data_set_info.get( "mention_list", None )
# 		name = normalize_string(name)
# 		name_words = word_tokenize(name)
# 		mention_info.append([name_words, data_set_id])
# 		for mention in mention_list:
# 			mention.encode('ascii','ignore')
# 			mention = normalize_string(mention).strip()
# 			mention = re.sub("\s\s+", " ", mention)
# 			if all(c.islower() for c in mention) and len(mention.split()) <= 2:
# 				continue    # to avoid pronoun mentions like 'data', 'time'
# 			sentences = split_into_sentences(mention)
# 			words = []
# 			for sentence in sentences:
# 				words += word_tokenize(sentence)
# 			words = [w for w in words if len(w)<15]
# 			if len(words) > 0  and len(words) <= 30:
# 				mention_info.append([words, data_set_id])
# logging.info("Captured mentions: {}.".format(len(mention_info)))

# vocab = get_vocab(mention_info)
# word2idx, idx2word = get_word2idx_idx2word(vocab)
# logging.info("Loading glove embeddings")
# glove_embeddings = get_embedding_matrix(word2idx, idx2word, args, normalization=False)
class_to_idx = {}
with open(args.tag_file_path, 'r') as f:
	for (n, i) in enumerate(f): #idx, label
		class_to_idx[i.strip()] = n

# if using_GPU:
# 	elmo = ElmoEmbedder("./elmo/options.json", "./elmo/weights.hdf5", 0)
# else:
# 	elmo = ElmoEmbedder("./elmo/options.json", "./elmo/weights.hdf5", -1)

# logging.info("embedd test data with glove and elmo vectors")
# embedded_rcc = []
# for example in tqdm(mention_info, total=len(mention_info)):
# 	embedded_rcc.append([embed_indexed_sequence(example[0], word2idx, glove_embeddings, elmo), class_to_idx[str(example[1])]])

# pickle.dump(embedded_rcc, open('./clf_training_embedd.data', "wb+"), protocol=-1)

with open('./clf_training_embedd.data', "rb+") as infile:
 embedded_rcc = pickle.load(infile)
import random
random.shuffle(embedded_rcc)
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

CNN_Text = CNN_Text(args.kernel_num, args.kernel_sizes, len(class_to_idx), 300 + 1024 + 4)
loss_criterion = nn.CrossEntropyLoss(reduction='elementwise_mean')
if using_GPU:
	CNN_Text = CNN_Text.cuda()
	state = torch.load(args.cnn_model_path)
else:
	state = torch.load(args.cnn_model_path, map_location='cpu')

if torch.cuda.device_count() > 1:
		CNN_Text= nn.DataParallel(CNN_Text)
		logging.info("Load CNN_Text Model")
		CNN_Text.load_state_dict(state['state_dict'])
CNN_Text.to(device)

optimizer = torch.optim.SGD(CNN_Text.parameters(), lr=args.lr, momentum=0.9, nesterov=True) #, weight_decay=1e-5)
#optimizer = torch.optim.Adam(CNN_Text.parameters(), lr=args.lr) #, weight_decay=1e-5, amsgrad=True)
optimizer.load_state_dict(state['optimizer'])
for param_group in optimizer.param_groups:
	param_group['lr'] = args.lr
best_val_loss = state['loss']
best_val_loss = 0
#prev_loss = best_val_loss

logging.info("*" * 50)
logging.info(CNN_Text)
logging.info("*" * 50)
sentences = [example[0] for example in embedded_rcc]
classes = [example[1] for example in embedded_rcc]

ten_foldsd = []
fold_size = int(len(embedded_rcc) / 10)
for i in range(10):
	ten_foldsd.append((sentences[i * fold_size:(i + 1) * fold_size], classes[i * fold_size:(i + 1) * fold_size]))

losses = []
accuracies = []
learning_rates = []
epochs = []

epoch_base = -args.num_epochs
num_iter = 0
CNN_Text.train()
logging.info("Ten Fold Validation, 1 fold size: {}".format(fold_size))
logging.info("Training with Batch Size: {}, Learning Rate: {}".format(args.batch_size, args.lr))
logging.info("Loaded best validation loss: {}".format(best_val_loss))
logging.info("*" * 50)
prev_train_loss = 100

try:
	while True:
		for i in range(10):
			logging.info("10 fold validation turn change...")
			first_in_fold = True
			training_sentences = []
			training_classes = []
			for j in range(10):
				if j != i:
					training_sentences.extend(ten_foldsd[j][0])
					training_classes.extend(ten_foldsd[j][1])
			training_dataset_rcc = TextDataset(training_sentences, training_classes)
			val_dataset_rcc = TextDataset(ten_foldsd[i][0], ten_foldsd[i][1])
			train_dataloader_rcc = DataLoader(dataset=training_dataset_rcc, batch_size=args.batch_size, shuffle=True, collate_fn=TextDataset.collate_fn)
			val_dataloader_rcc = DataLoader(dataset=val_dataset_rcc, batch_size=args.batch_size, shuffle=False, collate_fn=TextDataset.collate_fn)
			epoch_base += args.num_epochs
			for epoch in range(args.num_epochs):
				logging.info("Starting epoch {}".format(epoch + 1 + epoch_base))
				for (example_text, labels) in train_dataloader_rcc:
					example_text = Variable(example_text)
					labels = Variable(labels)
					if using_GPU:
						example_text = example_text.cuda()
						labels = labels.cuda()
					example_text = example_text.to(device)
					# predicted shape: (batch_size, seq_len, 2)
					logit = CNN_Text(example_text)
					#logit = F.softmax(logit, dim=1)
					loss = loss_criterion(logit, labels)
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
					num_iter += 1
					# Calculate validation and training set loss and accuracy every 200 gradient updates
					if num_iter % args.log_interval == 0:
						corrects = (torch.max(logit, 1)[1].view(labels.size()).data == labels.data).sum()
						batch_size = labels.size()[0]
						accuracy = 100.0 * float(corrects)/float(batch_size)
						logging.info('Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(num_iter,
																																							 loss.item(),
																																							 accuracy,
																																							 corrects,
																																							 batch_size))
						average_eval_loss, accuracy = evaluate_clf_cnn(val_dataloader_rcc, CNN_Text, loss_criterion, using_GPU)
						if best_val_loss > average_eval_loss:
							logging.info("New Best Validation Loss! {}, before: {}".format(average_eval_loss, best_val_loss))
							best_val_loss = average_eval_loss
							logging.info("Save Checkpoint to {}".format(args.cnn_model_path))
							torch.save({
													'state_dict': CNN_Text.state_dict(),
													'optimizer': optimizer.state_dict(),
													'loss': best_val_loss,
													}, args.cnn_model_path)
						elif best_val_loss == 0:
							logging.info("Initial Validation Loss! {}".format(average_eval_loss))
							best_val_loss = average_eval_loss
							logging.info("Save Checkpoint to {}".format(args.cnn_model_path))
							torch.save({
													'state_dict': CNN_Text.state_dict(),
													'optimizer': optimizer.state_dict(),
													'loss': best_val_loss,
													}, args.cnn_model_path)
						elif not first_in_fold and average_eval_loss >= prev_loss and loss.item() >= prev_train_loss:
							for param_group in optimizer.param_groups:
								prev_lr = param_group['lr']
								param_group['lr'] = prev_lr/1.1
								logging.info('gradient decay from {} to {}'.format(prev_lr, param_group['lr']))
						losses.append(average_eval_loss)
						accuracies.append(accuracy)
						temp_lr = 0
						for param_group in optimizer.param_groups:
							temp_lr = param_group['lr']
						learning_rates.append(temp_lr)
						epochs.append(epoch)
						prev_loss = average_eval_loss
						prev_train_loss = loss.item()
						first_in_fold = False
except KeyboardInterrupt:
	logging.info('timeout, dumping file to ./l_a_lr_ep_cnn.data')
	pickle.dump([losses, accuracies, learning_rates, epochs], open('./l_a_lr_ep.data', "wb+"), protocol=-1)




