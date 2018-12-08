#!/usr/bin/env bash
cd /project
pwd
# run python file
python3 -m spacy download en
echo "LC_ALL=en_US.UTF-8" >> /etc/environment
echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen
echo "LANG=en_US.UTF-8" > /etc/locale.conf
locale-gen en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
python3 ./test_parser.py
python3 ./make_abstract.py
python3 ./field_method.py
python3 ./inference.py # the classfier models are not trained enough.  
python3 ./make_citation_output.py
