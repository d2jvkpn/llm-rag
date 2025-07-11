#!/usr/bin/env python3
import json
from pathlib import Path
# os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = "True"

from src import embed, local_llms, process_doc
from src.utils import now

import litellm


def points_to_chunks(points):
    texts = []

    for p in points:
        chunk_id = p.payload['chunk_id']
        # print("~~~", p.playload)
        filename = repr(p.payload['filename'])

        text = p.payload['text']
        texts.append(f"chunk_id={chunk_id}, filename={filename}\n```text\n{text}\n```")

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


def rag_query_docs(user_input, docs, parameters):
    top_n = parameters['qdrant']['top_n']
    top_k = parameters['reranker']['top_k']

    for d in docs:
        embedding_doc(d, parameters['qdrant']['collection'])

    doc_ids = [d['doc_id'] for d in docs]

    vector = embed.litellm_embedding([user_input])[0]['data'][0]['embedding']
    hits = embed.search_doc(vector, doc_ids, top_n=top_n, score_threshold=0.5)
    print(f"{now()} rag_query_docs/search_doc: {len(hits.points)}")

    if len(hits.points) == 0:
        return []

    if not parameters['reranker']['enabled'] or len(hits.points) <= top_k:
        return points_to_chunks(hits.points)

    #for p in hits.points:
    #    chunk_id = p.payload['chunk_id']
    #    page = p.payload['page']
    #    filename = p.payload['filename']
    #    print(f"<-- hits: chunk_id={chunk_id}, page={page}, filename={filename}")
    #    #print(f"    text: {p.payload['text']}")

    texts = [p.payload['text'] for p in hits.points]
    print(f"{now()} rag_query_docs/rerank_texts: {top_k}")
    scores = local_llms.rerank_texts(user_input, texts)
    points = [p for _, p in sorted(zip(scores, hits.points), reverse=True)][:top_k]

    return points_to_chunks(points)


def call_llm(messages, parameters):
    provider, model = parameters['llm']['selected_model'].split("/", 1)
    temperature = parameters['llm']['temperature']

    print(f"{now()} call_llm: provider={provider}, model={model}, temperature={temperature}")

    found = next(
        (v for v in parameters['llm_models'] if v['provider'] == provider and v['model'] == model),
        None,
    )

    if found.get("hosted_vllm", False) is True:
        provider = "hosted_vllm"

    response = litellm.completion(
        custom_llm_provider=provider, model=model,
        api_base=found.get("api_base"), api_key=found.get("api_key"),
        max_tokens=parameters['llm']['max_tokens'], temperature=temperature,
        num_retries=3, timeout=60, stream=False,
        messages=messages,
    )

    return response
