#!/usr/bin/env python3
import os, uuid
from typing import Union

import yaml


Reranker, Embedding, Generator = None, None, None

def init_reranker(model_path):
    from sentence_transformers import CrossEncoder

    global Reranker

    Reranker = CrossEncoder(model_path)

def rerank_texts(query, texts):
    scores = Reranker.predict([(query, t) for t in texts])
    # output = [t for _, t in sorted(zip(scores, texts), reverse=True)][:10]

    return scores


def init_embedding(model_path):
    import torch
    from sentence_transformers import SentenceTransformer

    global Embedding, Dimension

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Embedding = SentenceTransformer(model_path)
    Embedding = Embedding.to(device)  # model.half()  # float16
    # dimension = Embedding.get_sentence_embedding_dimension()

def embeddding(content):
    vector = Embedding.encode(
        content,
        #  prompt="Represent this sentence for searching relevant passages: ",
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()

    return vector


def init_generator(model_path):
    import transformers, torch

    global Generator

    Generator = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={ "torch_dtype": torch.bfloat16 },
        device_map="auto", # auto, gpu, cpu
    )

# messages: `str`, `List[str]`, List[Dict[str, str]], or `List[List[Dict[str, str]]]`
def generator(messages, max_tokens):
    response = Generator(messages, max_new_tokens=max_tokens)
    return response[0]["generated_text"]
