import re
import time

sentence="dsaf CrimeStat II: A Spatial Statistics for the Analysis of Crime Incident Locations patial Statistic CrimeStat II: A Spatial Statistics Program for the Analysis of Crime Incident Locations patial Statistics Program for the Analysis of Crime Incident Locations patial Statistics Program for the Analysis of Crime Incident Locations"
sent = sentence.split()
print(sent[5:10])
print(sent[9])
print(sent[10])

def encode4(sub, sent):
	subwords, sentwords = sub.split(), sent.split()
	sub_pat = ('<MT-B>' + ' <MT-I>' * (len(subwords)-1)).strip()
	sent = sent.replace(sub, sub_pat)
	res = ['B' if w=='<MT-B>' else 'I' if w=='<MT-I>' else '_' for w in sent.split()]
	return res


# def encode(sub, sent):
#     subwords, sentwords = sub.split(), sent.split()
#     res = [0 for _ in sentwords]    
#     for i, word in enumerate(sentwords[:-len(subwords) + 1]):
#         if all(x == y for x, y in zip(subwords, sentwords[i:i + len(subwords)])):
#             for j in range(len(subwords)):
#                 res[i + j] = 1
#     return res

# def encode2(sub, sent):
# 	p = re.compile(sub)
# 	lst=[1 for i in range(len(sub.split()))]
# 	vect=[lst if items == '__match_word' else 0 for items in re.sub(p,'__match_word',sent).split()]
# 	vectlstoflst=[[vec] if isinstance(vec,int) else vec for vec in vect]
# 	flattened = [val for sublist in vectlstoflst for val in sublist]
# 	return flattened


# def label_substring(sentence, mention, prev_label_sequence, data_set_id):
#     label_sequence = prev_label_sequence
#     for i, word in enumerate(sentence[:-len(mention) + 1]):
#         if all(x == y for x, y in zip(mention, sentence[i:i + len(mention)])):
#             if label_sequence[i] is '_':
#                 label_sequence[i] = 'B-' + str(data_set_id)                
#             for j in range(len(mention)-1):
#                 label_sequence[i + j + 1] = 'I'
#     return label_sequence


# def encode3(sentences, mention_list):
# 	label_sequence_list = []
# 	for sentence in sentences:
# 		label_sequence = ['_' for _ in sentence]
# 		for data_set_id, mentions in mention_list:
# 			#print(data_set_id)
# 			for mention in mentions:
# 				label_sequence = label_substring(sentence, mention, label_sequence, data_set_id)
# 		label_sequence_list.append(label_sequence)
# 	return label_sequence_list



# search_word = 'CrimeStat II: A Spatial Statistics'
# sentence="dsaf CrimeStat II: A Spatial Statistics for the Analysis of Crime Incident Locations patial Statistic CrimeStat II: A Spatial Statistics Program for the Analysis of Crime Incident Locations patial Statistics Program for the Analysis of Crime Incident Locations patial Statistics Program for the Analysis of Crime Incident Locations"

# mention = search_word.split()
# sentences = sentence.split()
# start = time.time()

# start = time.time()
# for i in range(100000):
# 	encode(search_word, sentence)
# print(encode(search_word, sentence))
# spent = time.time() - start
# print(spent)

# start = time.time()
# for i in range(100000):
# 	encode2(search_word, sentence)
# print(encode2(search_word, sentence))
# spent = time.time() - start
# print(spent)

# start = time.time()
# for i in range(100000):
# 	encode3([sentences], [[1, [mention]]])
# print(encode3([sentences], [[1, [mention]]]))
# spent = time.time() - start
# print(spent)

# start = time.time()
# for i in range(100000):
# 	encode4(search_word, sentence)
# print(encode4(search_word, sentence))
# spent = time.time() - start
# print(spent)




