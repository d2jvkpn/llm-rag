#!/usr/bin/env python3


#### 1.
import nltk

# nltk.download('popular')
nltk.download('punkt')     # tokenizers/
nltk.download('stopwords') # corpora/
nltk.download('wordnet')   # corpora/
nltk.download('words')     # corpora/
nltk.download('maxent_ne_chunker') # chunkers
nltk.download('averaged_perceptron_tagger') # taggers

# nltk.download('punkt_tab')
# nltk.download(['book', 'all-corpora', 'all'])

nltk.data.path.append("./data/nltk_data")


#### 2.
from nltk.tokenize import sent_tokenize
from nltk import pos_tag

text = "Hello world. This is a test."
print(sent_tokenize(text))

print(sent_tokenize("This is a test sentence."))
print(pos_tag(["This", "is", "a", "test"]))


#### 3.
import markdown
from bs4 import BeautifulSoup
from langchain.schema import Document


def load_md_offline(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    html = markdown.markdown(raw)
    text = BeautifulSoup(html, "html.parser").get_text()
    return [Document(page_content=text)]


#### 4.
from langchain.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("README.md", mode="elements")
texts = loader.load()
