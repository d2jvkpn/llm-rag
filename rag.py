#!/usr/bin/env python3
import os, json, shutil
from pathlib import Path

from src import embed, local_llms, process_doc
from src.utils import now, file_md5


def copy_gradio_files(paths, dirctory):
    docs = []

    for p in paths:
        filename = Path(p).name
        doc_id = "md5-" + file_md5(p)
        #source_dir = os.path.dirname(p)
        #target_dir = os.path.join(dirctory, os.path.basename(source_dir))
        target_dir = Path(dirctory) / doc_id
        target_path = target_dir / filename
        # shutil.copy(p, save_path)
        doc = { "path": target_path, "doc_id": doc_id, "exists": False }

        if target_path.exists() and target_path.is_file():
            doc['exists'] = True
        else:
            os.makedirs(target_dir, exist_ok=True)
            print(f"{now()} copy_gradio_files: {p} -> {target_path}")
            shutil.copy(p, target_path)

        #if os.path.isdir(source_dir):
        #    print(f"{now()} remove duplicated: {p}")
        #    shutil.rmtree(source_dir)

        docs.append(doc)

    return docs


def points_to_chunks(points, max_filename_len=64):
    texts = []

    for p in points:
        chunk_id = p.payload['chunk_id']
        filename = Path(p.payload['path']).name
        if len(filename) > max_filename_len:
            filename = filename[:max_filename_len-3] + "..."

        text = p.payload['text']
        texts.append(f"chunk_id={chunk_id}, filename={repr(filename)}\n```text\n{text}\n```")

    return texts


# docs: {path: , doc_id: }, steps: document2chunks, litellm_embedding, vectordb_save
def embedding_doc(doc, collection):
    #doc_path = repr(doc['path'])
    ####
    if embed.vectordb_doc_exists(doc['doc_id']):
        print(f"{now()} embedding_doc/skip: {doc}")
        return

    doc_path = Path(doc['path']) # basename: doc_path.name
    doc_dir = doc_path.parent

    ####
    json_file = doc_dir / "doc_chunks.json"

    if json_file.exists():
        print(f"{now()} found doc_chunks file: {json_file}")

        with open(json_file, 'r', encoding="utf-8") as f:
            doc_chunks = json.load(f)
    else:
        print(f"{now()} call document2chunks: {doc}")
        doc_chunks = process_doc.document2chunks(str(doc['path']), doc['doc_id'])

        with open(json_file, 'w', encoding="utf-8") as f:
            json.dump(doc_chunks, f, ensure_ascii=False, indent=2)

    ####
    texts = [c['text']for c in doc_chunks['chunks']]
    json_file = doc_dir / f"embedding_responses.{collection}.json"

    if json_file.exists():
        print(f"{now()} found embedding_responses file: {json_file}")

        with open(json_file, 'r', encoding="utf-8") as f:
            embedding_responses = json.load(f)
    else:
        print(f"{now()} call litellm_embedding: chunks={len(texts)}, doc={doc}")
        embedding_responses = embed.litellm_embedding(texts)

        with open(json_file, 'w', encoding="utf-8") as f:
            json.dump(embedding_responses, f, ensure_ascii=False)

    vectors = []
    for response in embedding_responses:
        vectors.extend([v['embedding'] for v in response['data']])

    ####
    # TODO: ??atomicity
    print(f"{now()} embedding_doc/vectordb_save: chunks={len(texts)}, doc={doc}")
    embed.vectordb_save(doc_chunks, vectors, recreate=False)


def rag_query_docs(files, user_input, settings):
    top_n = settings['top_n']
    top_k = settings['top_k']

    if not files:
        return ([], [])

    paths = [v.name for v in files]
    docs = copy_gradio_files(paths, settings['upload_dir'])
    # print(f"{now()} 📎 Uploaded: {docs}")
    for d in docs:
        embedding_doc(d, settings['collection'])

    doc_ids = [d['doc_id'] for d in docs]

    vector = embed.litellm_embedding([user_input])[0]['data'][0]['embedding']
    hits = embed.search_doc(vector, doc_ids, top_n=top_n)
    print(f"{now()} rag_query_docs/search_doc: {len(hits.points)}")

    if len(hits.points) == 0:
        return (docs, [])

    if not settings['enabled'] or len(hits.points) <= top_k:
        return (docs, points_to_chunks(hits.points, 64))

    #for p in hits.points:
    #    chunk_id = p.payload['chunk_id']
    #    page = p.payload['page']
    #    path = p.payload['path']
    #    print(f"<-- hits: chunk_id={chunk_id}, page={page}, path={path}")
    #    #print(f"    text: {p.payload['text']}")

    texts = [p.payload['text'] for p in hits.points]
    print(f"{now()} rag_query_docs/rerank_texts: {top_k}")
    scores = local_llms.rerank_texts(user_input, texts)
    points = [p for _, p in sorted(zip(scores, hits.points), reverse=True)][:top_k]

    return (docs, points_to_chunks(points, 64))
