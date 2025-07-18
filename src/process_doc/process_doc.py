#!/usr/bin/env python3
import os, sys, re
sys.path.append(os.path.dirname(__file__))

import process_ooxml, process_opendoc
from process_utils import paragraphs_to_chunks, doc_filename

import ebooklib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from ebooklib import epub
from bs4 import BeautifulSoup


# doc_id=md5-xxxxxxxx
def document2chunks(path, doc_id, chunk_size=1000, chunk_overlap=100):
    ext = path.rsplit(".", 1)[-1]

    # TODO: odt, odp
    if ext == "odt":
        return process_opendoc.odt2chunks(path, doc_id)
    elif ext == "odp":
        return process_opendoc.odp2chunks(path, doc_id)
    elif ext == "docx":
        return process_ooxml.docx2chunks(path, doc_id)
    elif ext == "pptx":
        return process_ooxml.pptx2chunks(path, doc_id)
    elif ext == "pdf":
        return pdf2chunks(path, doc_id, chunk_size, chunk_overlap)
    elif ext == "md":
        return md2chunks(path, doc_id, chunk_size, chunk_overlap)
    elif ext == "epub":
        return epub2chunks(path, doc_id, chunk_size)
    elif ext == "txt":
        return text2chunks(path, doc_id, chunk_size)
    else:
        raise ValueError("unknown filetype")


def text2chunks(path, doc_id, chunk_size):
    filename = doc_filename(path)
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    #texts = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    paragraphs = [p.strip() for p in re.split(r'\n', text) if p.strip()]
    texts = paragraphs_to_chunks(paragraphs, chunk_size)

    chunks = []
    for i in range(len(texts)):
        payload = {
            "filename": filename, "doc_id": doc_id,
            "chunk_id": f"{doc_id}-page0-c{i}", "text": texts[i],
        }

        chunks.append(payload)

    meta = {
        "path": path, "doc_id": doc_id, "number_of_pages":  0,
        "chunk_size": 0, "chunk_overlap": 0, "number_of_chunks": len(chunks),
    }

    return { "meta": meta, "chunks": chunks }


def md2chunks(path, doc_id, chunk_size, chunk_overlap):
    filename = doc_filename(path)
    with open(path, encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.create_documents([text])

    chunks = []
    for i in range(len(docs)):
        payload = {
            "filename": filename, "doc_id": doc_id,
            "chunk_id": f"{doc_id}-page0-c{i}", "text": docs[i].page_content,
        }

        chunks.append(payload)

    meta = {
        "path": path, "doc_id": doc_id, "number_of_pages": 0,
        "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
        "number_of_chunks": len(chunks),
    }

    return { "meta": meta, "chunks": chunks }


def pdf2chunks(path, doc_id, chunk_size=1000, chunk_overlap=100):
    filename = doc_filename(path)
    number_of_pages = 0
    chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", "，", ",", " ", ""],
    )

    for page in PyPDFLoader(path).load_and_split():
        number_of_pages += 1
        texts = splitter.split_text(page.page_content)
        page = page.metadata['page']

        for i in range(len(texts)):
            payload = {
                "filename": filename, "doc_id": doc_id,
                "chunk_id": f"{doc_id}-page{page}-c{i}", "text": texts[i],
            }

            chunks.append(payload)

    meta = {
        "path": path, "doc_id": doc_id, "number_of_pages": number_of_pages,
        "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
        "number_of_chunks": len(chunks),
    }

    return { "meta": meta, "chunks": chunks }


def epub2chunks(path, doc_id, chunk_size=1000):
    def clean_text(text):
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    filename = doc_filename(path)
    book, chunks = epub.read_epub(path), []

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "xml")
            #tag = soup.find(['h1', 'h2', 'title'])
            #title = tag.get_text(strip=True) if title_tag else "Untitled Chapter"

            paragraphs = [
                clean_text(p.get_text())
                for p in soup.find_all("p") if p.get_text(strip=True)
            ]

            results = paragraphs_to_chunks(paragraphs, max_tokens=chunk_size)

            for i, chunk in enumerate(results):
                chunk = {
                    "filename": filename, "doc_id": doc_id,
                    "chunk_id": f"{doc_id}-page0-c{i}", "text": chunk,
                }
                chunks.append(chunk)

    meta = {
        "path": path, "doc_id": doc_id, "number_of_pages": 0,
        "chunk_size": chunk_size, "chunk_overlap": 0,
        "number_of_chunks": len(chunks),
    }

    return { "meta": meta, "chunks": chunks }
