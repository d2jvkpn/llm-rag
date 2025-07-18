#!/usr/bin/env python3

from process_utils import doc_filename

from odf import opendocument, text, draw


def odt2chunks(path, doc_id):
    filename = doc_filename(path)
    doc = opendocument.load(path)

    texts = [str(p) for p in doc.getElementsByType(text.P)]

    chunks = []
    for i in range(len(texts)):
        payload = {
            "filename": filename, "doc_id": doc_id,
            "chunk_id": f"{doc_id}-page0-c{i}", "text": texts[i],
        }

        chunks.append(payload)

    meta = {
        "path": path, "doc_id": doc_id, "number_of_pages": 0,
        "chunk_size": 0, "chunk_overlap": 0, "number_of_chunks": len(chunks),
    }

    return { "meta": meta, "chunks": chunks }

    # return "\n".join(p.firstChild.data for p in doc.getElementsByType(text.P) if p.firstChild)
    #content = read_odt_text("example.odt")
    #print(content)


def odp2chunks(path, doc_id):
    filename = doc_filename(path)
    doc = opendocument.load(path)
    slides = doc.getElementsByType(draw.Page)
    chunks = []

    for i, slide in enumerate(slides):
        elements = slide.getElementsByType(text.P)
        texts = [p.firstChild.data for p in elements if p.firstChild]

        payload = {
            "filename": filename, "doc_id": doc_id,
            "chunk_id": f"{doc_id}-page{i+1}-c0", "text": "\n".join(texts),
        }
        chunks.append(payload)

    meta = {
        "path": path, "doc_id": doc_id, "number_of_pages": len(slides),
        "chunk_size": 0, "chunk_overlap": 0, "number_of_chunks": len(chunks),
    }

    return { "meta": meta, "chunks": chunks }
