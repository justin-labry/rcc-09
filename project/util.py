from tqdm import tqdm
import torch
import numpy as np
import mmap
import ast
import csv
import re
from torch.utils.data import Dataset
import torch.nn as nn
from torch.autograd import Variable
import logging
from allennlp.commands.elmo import ElmoEmbedder
import json
import torch.nn.functional as F


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


def read_pub_json_files(args):
    pub_date_dict = dict()
    with open(args.train_pub_info_path) as json_publication_file:
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

    with open(args.test_pub_info_path) as json_publication_file:
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
    return pub_date_dict


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
                               'label_sequence': label_sequence})
        else:
            for sentence in sentences:
                label_sequence = ['_' for _ in sentence]
                output.append({'publication_id': publication_id,
                               'sentence': sentence,
                               'label_sequence': label_sequence})
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


# Misc helper functions
# Get the number of lines from a filepath
def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def get_pos2idx_idx2pos(vocab):
    word2idx = {}
    idx2word = {}
    for word in vocab:
        assigned_index = len(word2idx)
        word2idx[word] = assigned_index
        idx2word[assigned_index] = word
    return word2idx, idx2word


def index_sequence(item2idx, seq):
    embed = []
    for x in seq:
        embed.append(item2idx[x])
    assert (len(seq) == len(embed))
    return embed


def get_embedding_matrix(word2idx, idx2word, args, normalization=False):
    embedding_dim = 300
    glove_vectors = {}
    with open(args.glove_path) as glove_file:
        for line in tqdm(glove_file, total=get_num_lines(args.glove_path)):
            split_line = line.rstrip().split()
            word = split_line[0]
            if len(split_line) != (embedding_dim + 1) or word not in word2idx:
                continue
            assert (len(split_line) == embedding_dim + 1)
            vector = np.array([float(x) for x in split_line[1:]], dtype="float32")
            if normalization:
                vector = vector / np.linalg.norm(vector)
            assert len(vector) == embedding_dim
            glove_vectors[word] = vector

    logging.info("Number of pre-trained word vectors loaded: {}".format(len(glove_vectors)))

    # Calculate mean and stdev of embeddings
    all_embeddings = np.array(list(glove_vectors.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_stdev = float(np.std(all_embeddings))
    logging.info("Embeddings mean: {}".format(embeddings_mean))
    logging.info("Embeddings stdev: {}".format(embeddings_stdev))

    # Randomly initialize an embedding matrix of (vocab_size, embedding_dim) shape
    # with a similar distribution as the pretrained embeddings for words in vocab.
    vocab_size = len(word2idx)
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(embeddings_mean, embeddings_stdev)
    # Go through the embedding matrix and replace the random vector with a
    # pretrained one if available. Start iteration at 2 since 0, 1 are PAD, UNK
    for i in range(2, vocab_size):
        word = idx2word[i]
        if word in glove_vectors:
            embedding_matrix[i] = torch.FloatTensor(glove_vectors[word])
    if normalization:
        for i in range(vocab_size):
            embedding_matrix[i] = embedding_matrix[i] / float(np.linalg.norm(embedding_matrix[i]))
    embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    embeddings.weight = nn.Parameter(embedding_matrix)
    return embeddings


def get_vocab(raw_dataset):
    vocab = []
    for example in raw_dataset:
        vocab.extend(example[0])
    vocab = set(vocab)
    logging.info("vocab size: {}".format(len(vocab)))
    return vocab


def get_word2idx_idx2word(vocab):
    """

    :param vocab: a set of strings: vocabulary
    :return: word2idx: string to an int
             idx2word: int to a string
    """
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx2word = {0: "<PAD>", 1: "<UNK>"}
    for word in vocab:
        assigned_index = len(word2idx)
        word2idx[word] = assigned_index
        idx2word[assigned_index] = word
    return word2idx, idx2word


def embed_indexed_sequence(words, word2idx, glove_embeddings, elmo_embedder):
    indexed_sequence = [word2idx.get(x, 1) for x in words]
    # glove_part has shape: (seq_len, glove_dim)
    glove_part = glove_embeddings(Variable(torch.LongTensor(indexed_sequence)))
    capital_info = []
    for word in words:
        if all(c.islower() for c in word):
            capital_info.append([1, 0, 0, 0])
        elif all(c.isupper() for c in word):
            capital_info.append([0, 1, 0, 0])
        elif word[0].isupper():
            capital_info.append([0, 0, 1, 0])
        else:
            capital_info.append([0, 0, 0, 1])
    capital_tensor = Variable(torch.LongTensor(capital_info))
            #torch.LongTensor(indexed_sequence)
    # 2. embed the sequence by elmo vectors
    if elmo_embedder is not None:
        elmo_part = elmo_embedder.embed_sentence(words)[2]
        assert (elmo_part.shape == (len(words), 1024))
    if elmo_embedder is None:
        result = glove_part.data
    else:  # elmo != None, pos = None
        result = np.concatenate((glove_part.data, elmo_part, capital_tensor), axis=1)
    assert (len(words) == result.shape[0])
    return result


def evaluate(evaluation_dataloader, model, criterion, using_GPU):
    model.eval()

    # total_examples = total number of words
    total_examples = 0
    total_eval_loss = 0
    corrects = 0
    misses = 0
    #confusion_matrix = np.zeros((3, 3))
    for (eval_text, eval_lengths, eval_labels) in evaluation_dataloader:
        eval_text = Variable(eval_text)
        lengths = eval_lengths.numpy()
        eval_lengths = Variable(eval_lengths)
        eval_labels = Variable(eval_labels)
        if using_GPU:
            eval_text = eval_text.cuda()
            eval_lengths = eval_lengths.cuda()
            eval_labels = eval_labels.cuda()

        # predicted shape: (batch_size, seq_len, 3)
        with torch.no_grad():
            predicted = model(eval_text, eval_lengths)
        # Calculate loss for this test batch. This is averaged, so multiply
        # by the number of examples in batch to get a total.
        total_eval_loss += criterion(predicted.view(-1, 3), eval_labels.view(-1))
        # get 0 or 1 predictions
        # predicted_labels: (batch_size, seq_len)
        _, predicted_labels = torch.max(predicted.data, 2)
        total_examples += eval_lengths.size(0)
        for i in range(lengths.shape[0]):
            indexed_length = lengths[i]
            prediction = predicted_labels[i]
            label = eval_labels.data[i]
            for j in range(indexed_length):
                p = prediction[j]
                l = label[j]
                if p == l:
                    corrects += 1
                else:
                    misses += 1
    accuracy = float(corrects) / (float(corrects) + float(misses))
    average_eval_loss = total_eval_loss / evaluation_dataloader.__len__()

    # Set the model back to train mode, which activates dropout again.
    model.train()
    #print_info(confusion_matrix)
    return average_eval_loss.item(), accuracy


def evaluate_clf(evaluation_dataloader, model, criterion, using_GPU):
    model.eval()
    num_correct = 0
    total_examples = 0
    total_eval_loss = 0
    for (eval_text, eval_labels) in evaluation_dataloader:
        eval_text = Variable(eval_text)
        eval_labels = Variable(eval_labels)
        if using_GPU:
            eval_text = eval_text.cuda()
            eval_labels = eval_labels.cuda()
        with torch.no_grad():
            predicted = model(eval_text)
        # Calculate loss for this test batch. This is averaged, so multiply
        # by the number of examples in batch to get a total.
            total_eval_loss += criterion(predicted, eval_labels).item() * eval_labels.size(0)
        _, predicted_labels10 = torch.topk(predicted.data, 10, dim=1)
        for i in range(eval_labels.size(0)):
            predictions = predicted_labels10[i]
            prediction = predictions[-1]
            pub_date = eval_dates[i]
            for j in range(10):
                if predictions[j] not in dataset_id_to_date:
                    dataset_id_to_date[predictions[j]] = '1800'
                    prediction = predictions[j]
                    break
                elif pub_date >= int(dataset_id_to_date[predictions[j]]):
                    prediction = predictions[j]
                    break
            if prediction == eval_labels.data[i].item():
                num_correct += 1
        # _, predicted_labels = torch.max(predicted.data, 1)
        total_examples += eval_labels.size(0)
        # num_correct += torch.sum(predicted_labels == eval_labels.data)
    accuracy = float(num_correct) / float(total_examples)
    average_eval_loss = total_eval_loss / total_examples
    model.train()
    return average_eval_loss, accuracy, num_correct, total_examples


# def evaluate_clf_cnn(evaluation_dataloader, model, criterion, dataset_id_to_date, using_GPU):
#     model.eval()
#     corrects = 0
#     size = 0
#     total_eval_loss = 0
#     for example_text, example_classes, example_dates in evaluation_dataloader:
#         example_text = Variable(example_text)
#         example_classes = Variable(example_classes)
#         if using_GPU:
#             example_text = example_text.cuda()
#             example_classes = example_classes.cuda()
#         with torch.no_grad():
#             logit = model(example_text)
#             logit = F.softmax(logit, dim=1)
#             total_eval_loss += criterion(logit, example_classes).item() * example_classes.size(0)
#         _, predicted_labels10 = torch.topk(logit, 10, dim=1)
#         for i in range(example_classes.size(0)):
#             predictions = predicted_labels10[i]
#             prediction = predictions[-1]
#             pub_date = example_dates[i]
#             for j in range(10):
#                 if predictions[j] not in dataset_id_to_date:
#                     dataset_id_to_date[predictions[j]] = '1800'
#                     prediction = predictions[j]
#                     break
#                 elif pub_date >= int(dataset_id_to_date[predictions[j]]):
#                     prediction = predictions[j]
#                     break
#             if prediction == example_classes.data[i].item():
#                 corrects += 1
#         #corrects += (torch.max(logit, 1)[1].view(example_classes.size()).data == example_classes.data).sum()
#         size += example_classes.size()[0]
#     avg_loss = total_eval_loss / size
#     accuracy = 100.0 * float(corrects)/float(size)
#     logging.info('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(avg_loss,
#                                                                        accuracy,
#                                                                        corrects,
#                                                                        size))
#     model.train()
#     return avg_loss


def evaluate_clf_cnn(evaluation_dataloader, model, criterion, using_GPU):
    model.eval()
    corrects = 0
    size = 0
    total_eval_loss = 0
    for example_text, example_classes in evaluation_dataloader:
        example_text = Variable(example_text)
        example_classes = Variable(example_classes)
        if using_GPU:
            example_text = example_text.cuda()
            example_classes = example_classes.cuda()
        with torch.no_grad():
            logit = model(example_text)
            #logit = F.softmax(logit, dim=1)
            total_eval_loss += criterion(logit, example_classes).item() * example_classes.size(0)
        corrects += (torch.max(logit, 1)[1].view(example_classes.size()).data == example_classes.data).sum()
        size += example_classes.size()[0]
    avg_loss = total_eval_loss / size
    accuracy = 100.0 * float(corrects)/float(size)
    logging.info('Evaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    model.train()
    return avg_loss, accuracy


def update_confusion_matrix(matrix, predictions, labels, lengths):
    for i in range(lengths.shape[0]):
        indexed_length = lengths[i]
        prediction = predictions[i]
        label = labels[i]
        for j in range(indexed_length):
            p = prediction[j]
            l = label[j]
            matrix[p][l] += 1
    return matrix


def get_batch_predictions(predictions, lengths):
    pred_lst = []
    for i in range(lengths.shape[0]):  # each example i.e. each row
        indexed_length = lengths[i]
        prediction_padded = predictions[i]
        cur_pred_lst = []
        for j in range(indexed_length):  # inside each example: up to sentence length
            cur_pred_lst.append(prediction_padded[j].item())
        pred_lst.append(cur_pred_lst)
    return pred_lst


def write_predictions(raw_dataset, evaluation_dataloader, model, using_GPU, rawdata_filename):
    model.eval()

    predictions = []
    logging.info("Get predictions... takes long, we expect over 10 hours...")
    for (eval_text, eval_lengths, __) in tqdm(evaluation_dataloader):
        eval_text = Variable(eval_text)
        lengths = eval_lengths.numpy()
        eval_lengths = Variable(eval_lengths)
        if using_GPU:
            eval_text = eval_text.cuda()
            eval_lengths = eval_lengths.cuda()

        # predicted shape: (batch_size, seq_len, 2)
        with torch.no_grad():
            predicted = model(eval_text, eval_lengths)
        # get 0 or 1 predictions
        # predicted_labels: (batch_size, seq_len)
        _, predicted_labels = torch.max(predicted.data, 2)
        predictions.extend(get_batch_predictions(predicted_labels, lengths))

    # Set the model back to train mode, which activates dropout again.
    model.train()
    assert (len(predictions) == len(raw_dataset))

    # read original data
    data = []
    with open(rawdata_filename) as f:
        lines = csv.reader(f)
        for line in lines:
            data.append(line)

    # append predictions to the original data
    data[0].append('label_sequence')
    for i in range(len(predictions)):
        data[i + 1].append(predictions[i])
    return data


def print_info(matrix):
    precision = 100 * matrix[1, 1] / np.sum(matrix[1])
    recall = 100 * matrix[1, 1] / np.sum(matrix[:, 1])
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = 100 * (matrix[1, 1] + matrix[0, 0]) / np.sum(matrix)
    logging.info('Precision: {}, Recall: {}, F1: {}, Accuracy: {}'.format(precision, recall, f1, accuracy))


# Make sure to subclass torch.utils.data.Dataset
class TextDatasetWithGloveElmoSuffix(Dataset):
    def __init__(self, text, labels, word2idx, glove_embeddings, elmo):
        if len(text) != len(labels):
            raise ValueError("Differing number of sentences and labels!")
        # A list of numpy arrays, where each inner numpy arrays is sequence_length * embed_dim
        # embedding for each word is : glove + elmo + suffix
        self.text = text
        #  a list of list: each inner list is a sequence of 0, 1.
        # where each inner list is the label for the sentence at the corresponding index.
        self.labels = labels
        # Truncate examples that are longer than max_sequence_length.
        # Long sequences are expensive and might blow up GPU memory usage.
        self.word2idx = word2idx
        self.glove = glove_embeddings
        self.elmo = elmo

    def __getitem__(self, idx):
        example_text = self.text[idx]
        example_text = embed_indexed_sequence(example_text, self.word2idx, self.glove, self.elmo)
        example_label_seq = self.labels[idx]
        # Truncate the sequence if necessary
        example_length = example_text.shape[0]
        assert (example_length == len(example_label_seq))
        return example_text, example_length, example_label_seq

    def __len__(self):
        """
        Return the number of examples in the Dataset.
        """
        return len(self.labels)

    @staticmethod
    def collate_fn(batch):
        batch_padded_example_text = []
        batch_lengths = []
        batch_padded_labels = []

        # Get the length of the longest sequence in the batch
        max_length = -1
        for text, __, __ in batch:
            if len(text) > max_length:
                max_length = len(text)

        # Iterate over each example in the batch
        for text, length, label in batch:
            amount_to_pad = max_length - length
            # Tensor of shape (amount_to_pad,), converted to LongTensor
            pad_tensor = torch.zeros(amount_to_pad, text.shape[1])
            text = torch.Tensor(text)
            padded_example_text = torch.cat((text, pad_tensor), dim=0)

            # pad the labels with zero.
            padded_example_label = label + [0] * amount_to_pad

            # Add the padded example to our batch
            batch_padded_example_text.append(padded_example_text)
            batch_lengths.append(length)
            batch_padded_labels.append(padded_example_label)

        # Stack the list of LongTensors into a single LongTensor
        return (torch.stack(batch_padded_example_text),
                torch.LongTensor(batch_lengths),
                torch.LongTensor(batch_padded_labels))


# Make sure to subclass torch.utils.data.Dataset
class TextDatasetForClassfier_RNN(Dataset):
    def __init__(self, embedded_text, labels, pub_dates, max_sequence_length=100):
        if len(embedded_text) != len(labels):
            raise ValueError("Differing number of sentences and labels!")
        self.embedded_text = embedded_text
        self.labels = labels
        self.pub_dates = pub_dates
        self.max_sequence_length = max_sequence_length


    def __getitem__(self, idx):
        example_text = self.embedded_text[idx]
        example_label = self.labels[idx]
        example_pub_date = self.pub_dates[idx]
        # Truncate the sequence if necessary
        example_text = example_text[:self.max_sequence_length]
        example_length = example_text.shape[0]

        return example_text, example_length, example_label, example_pub_date

    def __len__(self):
        """
        Return the number of examples in the Dataset.
        """
        return len(self.labels)

    @staticmethod
    def collate_fn(batch):
        batch_padded_example_text = []
        batch_lengths = []
        batch_labels = []
        batch_dates = []

        # Get the length of the longest sequence in the batch
        max_length = max(batch, key=lambda example: example[1])[1]

        # Iterate over each example in the batch
        for text, length, label, pub_date in batch:
            amount_to_pad = max_length - length
            pad_tensor = torch.zeros(amount_to_pad, text.shape[1])
            text = torch.Tensor(text)
            padded_example_text = torch.cat((text, pad_tensor), dim=0)

            # Add the padded example to our batch
            batch_padded_example_text.append(padded_example_text)
            batch_lengths.append(length)
            batch_labels.append(label)
            batch_dates.append(pub_date)

        # Stack the list of LongTensors into a single LongTensor
        return (torch.stack(batch_padded_example_text),
                torch.LongTensor(batch_lengths),
                torch.LongTensor(batch_labels),
                batch_dates)


# Make sure to subclass torch.utils.data.Dataset
class TextDatasetForClassfier_CNN(Dataset):
    def __init__(self, embedded_text, classes):
        self.embedded_text = embedded_text
        self.classes = classes

    def __getitem__(self, idx):
        example_text = self.embedded_text[idx]
        example_class = self.classes[idx]
        # Truncate the sequence if necessary
        example_length = example_text.shape[0]

        return example_text, example_length, example_class

    def __len__(self):
        """
        Return the number of examples in the Dataset.
        """
        return len(self.classes)

    @staticmethod
    def collate_fn(batch):
        batch_padded_example_text = []
        batch_lengths = []
        batch_labels = []

        # Get the length of the longest sequence in the batch
        max_length = -1
        for text, __, __ in batch:
            if len(text) > max_length:
                max_length = len(text)

        # Iterate over each example in the batch
        for text, length, label in batch:
            amount_to_pad = max_length - length
            pad_tensor = torch.zeros(amount_to_pad, text.shape[1])
            text = torch.Tensor(text)
            padded_example_text = torch.cat((text, pad_tensor), dim=0)

            # Add the padded example to our batch
            batch_padded_example_text.append(padded_example_text)
            batch_labels.append(label)

        # Stack the list of LongTensors into a single LongTensor
        return (torch.stack(batch_padded_example_text),
                torch.LongTensor(batch_labels))

# Make sure to subclass torch.utils.data.Dataset
class TextDatasetForClassfier_CNN_ForTest(Dataset):
    def __init__(self, embedded_text, pub_ids, pub_dates, word_seqs):
        self.embedded_text = embedded_text
        self.pub_ids = pub_ids
        self.pub_dates = pub_dates
        self.word_seqs = word_seqs

    def __getitem__(self, idx):
        example_text = self.embedded_text[idx]
        example_id = self.pub_ids[idx]
        example_pub_date = self.pub_dates[idx]
        example_word_seq = self.word_seqs[idx]
        # Truncate the sequence if necessary
        example_length = example_text.shape[0]

        return example_text, example_length, example_id, example_pub_date, example_word_seq

    def __len__(self):
        """
        Return the number of examples in the Dataset.
        """
        return len(self.pub_ids)

    @staticmethod
    def collate_fn(batch):
        batch_padded_example_text = []
        batch_lengths = []
        batch_dates = []
        batch_pub_ids = []
        batch_word_seqs = []

        # Get the length of the longest sequence in the batch
        max_length = -1
        for text, __, __, __, __ in batch:
            if len(text) > max_length:
                max_length = len(text)

        # Iterate over each example in the batch
        for text, length, pub_id, pub_date, word_seq in batch:
            amount_to_pad = max_length - length
            pad_tensor = torch.zeros(amount_to_pad, text.shape[1])
            text = torch.Tensor(text)
            padded_example_text = torch.cat((text, pad_tensor), dim=0)

            # Add the padded example to our batch
            batch_padded_example_text.append(padded_example_text)
            batch_pub_ids.append(pub_id)
            batch_dates.append(pub_date)
            batch_word_seqs.append(word_seq)

        # Stack the list of LongTensors into a single LongTensor
        return (torch.stack(batch_padded_example_text),
                batch_pub_ids,
                batch_dates,
                batch_word_seqs)



class TextDatasetWithGloveElmoSuffix_ForTest(Dataset):
    def __init__(self, pub_id, embedded_text):
        self.pub_id = pub_id
        self.embedded_text = embedded_text

    def __getitem__(self, idx):
        example_text = self.embedded_text[idx]
        example_pub_id = self.pub_id[idx]
        # Truncate the sequence if necessary
        example_length = example_text.shape[0]
        return example_text, example_length, example_pub_id

    def __len__(self):
        """
        Return the number of examples in the Dataset.
        """
        return len(self.pub_id)

    @staticmethod
    def collate_fn(batch):
        batch_padded_example_text = []
        batch_lengths = []
        batch_pub_ids = []

        # Get the length of the longest sequence in the batch
        max_length = -1
        for text, __, __ in batch:
            if len(text) > max_length:
                max_length = len(text)

        # Iterate over each example in the batch
        for text, length, pub_id in batch:
            amount_to_pad = max_length - length
            # Tensor of shape (amount_to_pad,), converted to LongTensor
            pad_tensor = torch.zeros(amount_to_pad, text.shape[1])
            text = torch.Tensor(text)
            padded_example_text = torch.cat((text, pad_tensor), dim=0)

            # Add the padded example to our batch
            batch_padded_example_text.append(padded_example_text)
            batch_lengths.append(length)
            batch_pub_ids.append(pub_id)

        # Stack the list of LongTensors into a single LongTensor
        return (torch.stack(batch_padded_example_text),
                torch.LongTensor(batch_lengths),
                batch_pub_ids)



