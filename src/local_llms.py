#!/usr/bin/env python3


Reranker, Embedding, Generator, Pretrained = None, None, None, None


#### 1. reranker
def init_reranker(model_path):
    from sentence_transformers import CrossEncoder

    global Reranker

    Reranker = CrossEncoder(model_path)

def rerank_texts(query, texts):
    scores = Reranker.predict([(query, t) for t in texts])
    # output = [t for _, t in sorted(zip(scores, texts), reverse=True)][:10]

    return scores


#### 2. embedding
def init_embedding(model_path):
    import torch
    from sentence_transformers import SentenceTransformer

    global Embedding

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Embedding = SentenceTransformer(model_path)
    Embedding = Embedding.to(device)  # model.half()  # float16
    # dimension = Embedding.get_sentence_embedding_dimension()

def embeddding(texts: list[str]): # Union[str, list[str]]
    vectors = Embedding.encode(
        texts,
        #  prompt="Represent this sentence for searching relevant passages: ",
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()

    return vectors


#### 3. generator
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
    return response[0]['generated_text']


#### 4. pretrained
def init_pretrained(model_path):
    """ model_path
    ├── config.json
    ├── pytorch_model.bin or model.safetensors
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── vocab.txt 或 merges.txt / special_tokens_map.json
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    global Pretrained

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    Pretrained = (tokenizer, model)

def pretrained(input_text: str, max_tokens):
    tokenizer, model = Pretrained

    #### encode using chat template
    #input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
    #output = model.generate(input_ids, max_new_tokens=max_tokens)

    inputs = tokenizer(input_text, return_tensors="pt")
    # {
    #   'input_ids': tensor([[...]]),
    #   'attention_mask': tensor([[...]]),
    #   ?? 'token_type_ids': tensor([[0, ..., 0, ..., 1]]),
    # }

    outputs = model.generate(**inputs, max_new_tokens=max_tokens)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
