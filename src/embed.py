#!/usr/bin/env python3
import os, uuid
# from typing import Union
os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = "True"

from .chrono import Chrono
from .utils import now

import requests, litellm # yaml
from qdrant_client import QdrantClient, models as qmodels
# TODO: langdetect


QConf, QClient = {}, None
EmbeddingConfig = {}

def init(config): # yaml filepath
    global QConf, QClient, EmbeddingConf

    #with open(filepath, 'r') as f:
    #    Config = yaml.safe_load(f)
    QConf = config['qdrant']
    EmbeddingConf = config['embedding']

    QClient = QdrantClient(
        url=QConf['addr'], prefer_grpc=True, https=False, timeout=30,
    )


def embedding_api(texts: list[str]): # Union[str, list[str]]
    api_base = EmbeddingConf['api_base']     # "http://127.0.0.1:11434/api/embed"
    api_key = EmbeddingConf.get('api_key', '')
    model = EmbeddingConf['model'] # "bge-m3:567m"

    headers = { "Content-Type": "application/json", "Authorization": f"Bearer {api_key}" }
    data = { "model": model, "encoding_format": "float", "input": texts }

    response = requests.post(api_base+"/v1/embeddings", headers=headers, json=data)

    if response.status_code == 200:
        ans = response.json()
        return [v['embedding'] for v in ans['data']]
    else:
        raise Exception(f"API Error: {response.status_code}, {response.text}")


def litellm_embedding(texts: list[str]): # Union[str, list[str]], "hello", ['hello", "world']
    api_base = EmbeddingConf['api_base']         # "http://127.0.0.1:11434/api/embed"
    api_key = EmbeddingConf.get('api_key', '')
    model = EmbeddingConf['model']               # "bge-m3:567m"

    if EmbeddingConf.get("hosted_vllm", False):
        provider = "hosted_vllm"
    else:
        provider = EmbeddingConf['provider']     # ollama


    batches = [texts[i : i + 10] for i in range(0, len(texts), 10)]
    vectors = []

    for text in batches:
        response = litellm.embedding(
            model, custom_llm_provider=provider,
            api_base=api_base, api_key=api_key,
            input=text,
        )

        vectors.extend([v['embedding'] for v in response['data']])

    return vectors


def vectordb_doc_exists(doc_id):
    collection = QConf['collection']

    if not QClient.collection_exists(collection):
        return False

    selector = qmodels.Filter(must=[
        qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id)),
    ])

    hits = QClient.scroll(collection_name=collection, scroll_filter=selector, limit=1)
    return len(hits[0]) > 0

def vectordb_save(doc, vectors, recreate=False):
    assert(len(doc['chunks']) == len(vectors))
    t0 = Chrono()
    collection = QConf['collection']
    dimension = len(vectors[0])

    # print(f"==> {Chrono()} Starting embedding_doc: doc={doc['meta']}")
    # client.delete_collection(collection)
    optimizers_config = qmodels.OptimizersConfigDiff(
        indexing_threshold=0, memmap_threshold=20000,
    )

    field_schema = qmodels.TextIndexParams(
        type="text", tokenizer="word", min_token_len=2, max_token_len=20,
    )

    if not QClient.collection_exists(collection):
        vectors_config = qmodels.VectorParams(
            size=dimension,
            distance=qmodels.Distance.COSINE,
            on_disk=True,
        )

        QClient.create_collection(
            collection_name=collection,
            vectors_config=vectors_config,
            # indexing_threshold: 立即索引所有向量, memmap_threshold: 超过20k向量使用mmap
            optimizers_config=optimizers_config,
            # m: 每个节点的连接数, ef_construct: 构建时的候选集大小
            hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=100),
        )

        QClient.create_payload_index(
            collection_name=collection, field_name="doc_id", field_schema=field_schema,
        )

    doc_id = doc['meta']['doc_id']
    selector = qmodels.Filter(must=[
        qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id)),
    ])

    if recreate and vectordb_doc_exists(doc_id) :
        QClient.delete(collection_name=collection, points_selector=selector)

    doc['meta'].update({ "collection": collection })

    #print(f"--> {Chrono()} upsert to the vector database")
    points = [
        qmodels.PointStruct(id=str(uuid.uuid4()), vector=vectors[i], payload=doc['chunks'][i])
        for i in range(len(vectors))
    ]

    QClient.upsert(collection_name=collection, points=points)

    doc['meta'].update({ "embedding_at": f"{t0}", "embedding_elapsed": t0.elapsed() })

    #print(f"<== {Chrono()} Done")
    #print(f"==> {Chrono()} process_doc 4")
    return doc


def search_doc(vector, doc_ids, top_n=10, score_threshold=0.5):
    collection = QConf['collection']

    #query_filter=qmodels.Filter(
    #    must=[qmodels.FieldCondition(key="category", match=qmodels.MatchValue(value="technology"))],
    #),

    ## hnsw_ef: 搜索时的候选集大小, exact: 使用近似搜索
    #search_params=models.SearchParams(hnsw_ef=128, exact=False)

    filters = [
        qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))
        for doc_id in doc_ids
    ]

    hits = QClient.query_points(
        collection_name=collection, query=vector,
        #prefetch = models.Prefetch(query=[1, 23, 45, 67], using="mrl_byte", limit=1000),
        #search_params=search_params,
        query_filter=qmodels.Filter(must=filters),
        limit=top_n, with_payload=True, offset=None, score_threshold=score_threshold,
    )

    #context = "\n---\n".join([point.payload['text'].strip() for point in hits.points])
    #texts = [point.payload['text'].strip() for point in hits.points]

    return hits


# docs: [{path: , doc_id: }]
def embedding_docs(docs, document2chunks):
    for d in docs:
        doc_path = repr(d['path'])

        if vectordb_doc_exists(d['doc_id']):
            print(f"{now()} ---> embedding_docs skip: {doc_path}")
            continue

        print(f"{now()} ---> document2chunks: {d}")
        doc = document2chunks(d['path'], d['doc_id'])

        texts = [c['text']for c in doc['chunks']]
        print(f"{now()} ---> litellm_embedding: chunks={len(texts)}, doc_path={doc_path}")
        vectors = litellm_embedding(texts)

        print(f"{now()} ---> vectordb_save: chunks={len(texts)}, doc_path={doc_path}")
        vectordb_save(doc, vectors, recreate=False)


def points_to_chunks(points, max_filename_len=64):
    texts = []

    for p in points:
        chunk_id = p.payload['chunk_id']
        filename = os.path.basename(p.payload['path'])
        if len(filename) > max_filename_len:
            filename = filename[:max_filename_len-3] + "..."

        text = p.payload['text']
        texts.append(f"chunk_id={chunk_id}, filename={repr(filename)}\n```text\n{text}\n```")

    return texts
