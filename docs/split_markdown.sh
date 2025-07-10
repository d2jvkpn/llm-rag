#!/bin/bash
set -eu -o pipefail; _wd=$(pwd); _dir=$(readlink -f `dirname "$0"`)


pip3 install langchain langchain-community "unstructured[md]"
pip3 install markdown lxml html5lib

exit

mkdir -p ./data/nltk_data/tokenizers
wget -P data https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip
unzip data/punkt.zip -d ./data/nltk_data/tokenizers

https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger_eng.zip
https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip
