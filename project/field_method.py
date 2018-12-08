import pke
import json 
from tqdm import tqdm
from allennlp.commands.elmo import ElmoEmbedder
import scipy
import nltk
import logging
import torch

nltk.download('stopwords')

PUB_PATH = '../data/input/'
JSON_PATH = './formatted-data/'
ABSTRACT_DATA_PATH = './formatted-data/abstracts/'
ELMO_PATH = './elmo/'
JSON_WRITE_PATH = '../data/output/'


def find_field(JSON_PATH, ABSTRACT_DATA_PATH, ELMO_PATH):
	with open(PUB_PATH + 'publications.json') as json_publication_file, open(JSON_PATH + 'sage_research_fields.json') as json_field_file, open(JSON_PATH + 'sage_research_methods.json') as json_method_file, open(JSON_WRITE_PATH + 'research_fields.json', 'w') as field_outfile, open(JSON_WRITE_PATH + 'methods.json', 'w') as method_outfile:
		
		publication_list = json.load(json_publication_file)
		field_list = json.load(json_field_file)
		method_list = json.load(json_method_file)
		using_GPU = torch.cuda.is_available()
		if using_GPU:
			elmo = ElmoEmbedder(ELMO_PATH + 'options.json', ELMO_PATH + 'weights.hdf5', 0)
		else:
			elmo = ElmoEmbedder(ELMO_PATH + 'options.json', ELMO_PATH + 'weights.hdf5', -1)
		
		field_json_list = []
		method_json_list = []

		logging.info("Extract research_fields and methods...")
		for publication_info in tqdm(publication_list, total=len(publication_list)):
			field_result = dict()
			method_result = dict()
			publication_id = publication_info.get("publication_id", None)
			text_file_name = publication_info.get("text_file_name", None)
			#print("\n\nLOAD : %s\n" % text_file_name)		
			#print("LOAD : %s\n" % text_file_name)
			extractor = pke.unsupervised.TopicRank()

			extractor.load_document(input= ABSTRACT_DATA_PATH + text_file_name, language='en')

			extractor.candidate_selection()

			extractor.candidate_weighting()

			keyphrases = extractor.get_n_best(n=20)
			
			embed_key = []
			for key in keyphrases:
				#print(key)
				tokens = key[0].split(' ')
				vectors = elmo.embed_sentence(tokens)
				embed_key.append(vectors[2])

				
			field_info_min = 1000
			field_info_index = ''
			for field_info in field_list:
				tokens = field_info.split(' ')
				vectors = elmo.embed_sentence(tokens)[2]
				for key in embed_key:
					for token_key in key:
						for word in vectors:
							temp = scipy.spatial.distance.cosine(word, token_key)
							if field_info_min > temp:
								field_info_min = temp
								field_info_index = field_info
			#print(field_info_index)


			detail_info_min = 1000
			detail_info_index = ''
			for detail_info in field_list[field_info_index]:
				tokens = detail_info.split(' ')
				vectors = elmo.embed_sentence(tokens)[2]
				for key in embed_key:
					for token_key in key:
						for word in vectors:
							temp = scipy.spatial.distance.cosine(word, token_key)
							if detail_info_min > temp:
								detail_info_min = temp
								detail_info_index = detail_info
			#print(detail_info_index)



			ddetail_info_min = 1000
			ddetail_info_index = ''
			for ddetail_info in field_list[field_info_index][detail_info_index]:
				tokens = ddetail_info['fieldAltLabel'].split(' ')
				vectors = elmo.embed_sentence(tokens)[2]
				for key in embed_key:
					for token_key in key:
						for word in vectors:
							temp = scipy.spatial.distance.cosine(word, token_key)
							if ddetail_info_min > temp:
								ddetail_info_min = temp
								ddetail_info_index = ddetail_info['fieldAltLabel']
			

			field_result["publication_id"] = publication_id
			field_result["research_field"] = ddetail_info_index
			field_result["score"] = round(1 - (field_info_min + detail_info_min + ddetail_info_min)/3, 3)
			field_json_list.append(field_result)
			#print(field_json_list)
			#print(field_result)	
			#print("FIELDS : %s\n" % ddetail_info_index)	



			keyphrases = extractor.get_n_best(n=5)
			
			embed_key = []
			for key in keyphrases:
				#print(key)
				tokens = key[0].split(' ')
				vectors = elmo.embed_sentence(tokens)
				embed_key.append(vectors[2])



			method_info_min = 1000
			method_info_index = ''
			count = 0
			for method_info in method_list['@graph']:
				if method_info['@id'] == "_:N872c4d9408ca446eb0e1391e1bfbddcb":
					continue
				else:
					if count == 150:
						#print("COUNT_CUT!!\n")
						count = 0
						break;
					else:
						tokens = method_info['skos:prefLabel']['@value'].split(' ')
						#print(tokens)
						vectors = elmo.embed_sentence(tokens)[2]
						for key in embed_key:
							for token_key in key:
								for word in vectors:
									temp = scipy.spatial.distance.cosine(word, token_key)
									if method_info_min > temp:
										method_info_min = temp
										method_info_index = method_info['skos:prefLabel']['@value']
				
				if method_info_index != method_info['skos:prefLabel']['@value']:
					count += 1
		
	
			method_result["publication_id"] = publication_id
			method_result["method"] = method_info_index
			method_result["score"] = round(1- method_info_min, 3)
			method_json_list.append(method_result)
			#print(method_result)
			#print("METHODS : %s\n" % method_info_index)
		json.dump(field_json_list, field_outfile, indent =4)
		json.dump(method_json_list, method_outfile, indent =4)



logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
find_field(JSON_PATH, ABSTRACT_DATA_PATH, ELMO_PATH)









