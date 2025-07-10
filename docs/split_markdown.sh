#!/bin/bash
set -eu -o pipefail; _wd=$(pwd); _dir=$(readlink -f `dirname "$0"`)


pip3 install langchain langchain-community "unstructured[md]"
pip3 install markdown lxml html5lib

exit

# mkdir -p ./data/nltk_data/tokenizers
# wget -P data https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip
# unzip data/punkt.zip -d ./data/nltk_data/tokenizers

# https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger_eng.zip
# https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip


_py_nltk = """python
import nltk

####
# nltk.download('popular')
nltk.download('punkt')     # tokenizers/
nltk.download('stopwords') # corpora/
nltk.download('wordnet')   # corpora/
nltk.download('words')     # corpora/
nltk.download('maxent_ne_chunker') # chunkers
nltk.download('averaged_perceptron_tagger') # taggers

# nltk.download('punkt_tab')
# nltk.download(['book', 'all-corpora', 'all'])


####
nltk.data.path.append("./data/nltk_data")

####
from nltk.tokenize import sent_tokenize
from nltk import pos_tag

text = "Hello world. This is a test."
print(sent_tokenize(text))


print(sent_tokenize("This is a test sentence."))
print(pos_tag(["This", "is", "a", "test"]))
"""


_py_md = """python
import markdown
from bs4 import BeautifulSoup
from langchain.schema import Document

def load_md_offline(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    html = markdown.markdown(raw)
    text = BeautifulSoup(html, "html.parser").get_text()
    return [Document(page_content=text)]
"""


_py_md2chunks = """python
from langchain.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader(path, mode="elements")
texts = loader.load()
"""

