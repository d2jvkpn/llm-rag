#!/usr/bin/env python3
import json
from pathlib import Path

#import os, sys
#sys.path.append(os.path.dirname(__file__))
#import embed, local_llms, process_doc

from . import embed, local_llms
from .process_doc import process_doc
from .utils import now


def points_to_chunks(points):
    texts = []

    for p in points:
        d = p.payload
        texts.append("chunk_id={}, score={:.3f}, filename={}\n```text\n{}\n```".format(
            d['chunk_id'], p.score, repr(d['filename']), d['text'],
        ))

    return texts


def embeding_tokens_usage(responses):
    usage = [0, 0, 0]

    for r in responses:
        usage[0] += r['usage']['prompt_tokens']
        usage[1] += r['usage']['completion_tokens']
        usage[2] += r['usage']['total_tokens']

    return { "prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": usage[2] }


# docs: { path: , doc_id: }, steps: document2chunks, litellm_embedding, vectordb_save
def embedding_doc(doc):
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
    json_file = doc_dir / f"embedding_responses.{embed.QConf['collection']}.json"

    if json_file.exists():
        print(f"{now()} found embedding_responses file: {json_file}")

        with open(json_file, 'r', encoding="utf-8") as f:
            embedding_responses = json.load(f)
    else:
        print(f"{now()} call litellm_embedding: chunks={len(texts)}, doc={doc}")
        embedding_responses = embed.litellm_embedding(texts)

        with open(json_file, 'w', encoding="utf-8") as f:
            json.dump(embedding_responses, f, ensure_ascii=False)

        usage = embeding_tokens_usage(embedding_responses)
        print(f"{now()} <-- embedding tokens usage: {usage}")

    vectors = []
    for response in embedding_responses:
        vectors.extend([v['embedding'] for v in response['data']])

    ####
    # TODO: ??atomicity
    print(f"{now()} embedding_doc/vectordb_save: chunks={len(texts)}, doc={doc}")
    embed.vectordb_save(doc_chunks, vectors, recreate=False)


def rag_query_docs(user_input, docs, parameters):
    top_n = parameters['qdrant']['top_n']
    limit = top_n*2 if parameters['reranker']['enabled'] else top_n

    for d in docs:
        embedding_doc(d)

    doc_ids = [d['doc_id'] for d in docs]

    vector = embed.litellm_embedding([user_input])[0]['data'][0]['embedding']
    hits = embed.search_doc(
        vector, doc_ids, top_n=limit,
        score_threshold=parameters['qdrant']['score_threshold'],
    )

    print("{} rag_query_docs/search_doc: limit={}, top_n={}, hits={}".format(
        now(), limit, top_n, len(hits.points),
    ))

    if len(hits.points) == 0:
        return []

    if not parameters['reranker']['enabled'] or len(hits.points) <= top_n:
        return points_to_chunks(hits.points)

    #for p in hits.points:
    #    chunk_id = p.payload['chunk_id']
    #    page = p.payload['page']
    #    filename = p.payload['filename']
    #    print(f"<-- hits: chunk_id={chunk_id}, page={page}, filename={filename}")
    #    #print(f"    text: {p.payload['text']}")

    texts = [p.payload['text'] for p in hits.points]
    print(f"{now()} rag_query_docs/rerank_texts: {top_n}")

    scores = local_llms.rerank_texts(user_input, texts)
    points = [p for _, p in sorted(zip(scores, hits.points), reverse=True)][:top_n]

    return points_to_chunks(points)
<<<<<<< HEAD


def call_llm(messages, parameters):
    provider, model = parameters['llm']['selected_model'].split("/", 1)
    temperature = parameters['llm']['temperature']
    llm_models = parameters['llm_models']
<<<<<<< HEAD

    if found.get("hosted_vllm", False) is True:
        provider = "hosted_vllm"
=======
>>>>>>> 9f8d5e8 (...)

    print(f"{now()} call_llm: provider={provider}, model={model}, temperature={temperature}")

    found = next(
        (v for v in llm_models if v['provider'] == provider and v['model'] == model),
        None,
    )

    response = litellm.completion(
        custom_llm_provider=provider, model=model,
        api_base=found.get("api_base"), api_key=found.get("api_key"),
        max_tokens=parameters['llm']['max_tokens'], temperature=temperature,
        num_retries=3, timeout=60, stream=False,
        messages=messages,
    )

    return response
=======
>>>>>>> 151d1d9 (...)
