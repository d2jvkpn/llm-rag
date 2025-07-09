#!/usr/bin/env python3
import re

import docx, pptx
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# doc_id=md5-xxxxxxxx
def document2chunks(path, doc_id, chunk_size=1000, chunk_overlap=100):
    ext = path.rsplit(".", 1)[-1]
    number_of_pages = 0

    if ext == "pptx":
       return pptx2chunks(path, doc_id)
    elif ext == "pdf":
        return pdf2chunks(path, doc_id, chunk_size, chunk_overlap)
    elif ext == "txt":
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        texts = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    elif ext == "docx":
       doc = docx.Document(path)
       texts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    else:
        raise ValueError("unknown filetype")

    chunks = []
    for i in range(len(texts)):
        payload = {
            "path": path, "doc_id": doc_id,
            "chunk_id": f"{doc_id}-page0-c{i}", "text": texts[i],
        }

        chunks.append(payload)

    meta = {
        "path": path, "doc_id": doc_id, "number_of_pages": number_of_pages,
        "chunk_size": 0, "chunk_overlap": 0, "number_of_chunks": len(chunks),
    }

    return {"chunks": chunks, "meta": meta}


def pptx2chunks(path, doc_id):
    prs = pptx.Presentation(path)
    chunks = []
    number_of_pages = len(prs.slides)

    for i, slide in enumerate(prs.slides):
        paragraphs = []
        page = i+1

        for shape in slide.shapes:
            if not shape.has_text_frame: continue
            for paragraph in shape.text_frame.paragraphs:
                text = paragraph.text.strip()
                if text: paragraphs.append(text)

        if len(paragraphs) == 0: continue

        payload = {
            "path": path, "doc_id": doc_id,
            "chunk_id": f"{doc_id}-page{page}-c{i}", "text": "\n".join(paragraphs),
        }
        chunks.append(payload)

    meta = {
        "path": path, "doc_id": doc_id, "number_of_pages": number_of_pages,
        "chunk_size": 0, "chunk_overlap": 0, "number_of_chunks": len(chunks),
    }

    return { "chunks": chunks, "meta": meta }


def pdf2chunks(path, doc_id, chunk_size=1000, chunk_overlap=100):
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
                "path": path, "doc_id": doc_id,
                "chunk_id": f"{doc_id}-page{page}-c{i}", "text": texts[i],
            }

            chunks.append(payload)

    meta = {
        "path": path, "doc_id": doc_id, "number_of_pages": number_of_pages,
        "chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "number_of_chunks": len(chunks),
    }

    return { "chunks": chunks, "meta": meta }
