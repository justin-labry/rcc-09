# imports
import codecs
import json
import shutil


# declare variables
publications_json_path = None
json_publication_file = None
publication_list = None
publication_counter = -1
publication_info = None
pub_date = None
unique_identifier = None
text_file_name = None
pdf_file_name = None
title = None
publication_id = None
citation_file_from = None
citation_file_to = None

# set path to publications.json
publications_json_path = "../data/input/publications.json"
training_publications = [] # id, title, text_file_name, pub_date

# open the publications.json file
with open(publications_json_path) as json_publication_file:

    # parse it as JSON
    publication_list = json.load(json_publication_file)

    # loop over the elements in the list
    publication_counter = 0
    for publication_info in publication_list:

        # increment counter
        publication_counter += 1

        # get information on publication:
        publication_id = publication_info.get( "publication_id", None )
        title = publication_info.get( "title", None )
        text_file_name = publication_info.get( "text_file_name", None )
        unique_identifier = publication_info.get( "unique_identifier", None ) # id가 bbk로 시작하면 pub_date은 None임
        if 'bbk' not in unique_identifier:
            pub_date = publication_info.get( "pub_date", None )
        else:
            pub_date = 'NA'
        training_publications.append([publication_id, title, text_file_name, pub_date])

for t in training_publications:
    print(t)

