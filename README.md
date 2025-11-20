# LLM-RAG
---
```meta
date: 2025-07-10
authors: []
version: 0.1.2
```

This project is a Retrieval-Augmented Generation (RAG) application built with Python 3, designed to 
provide accurate, context-aware answers by combining large language models with an efficient 
document retrieval pipeline. The system integrates a user-friendly Gradio WebUI, making it easy for 
end users to upload documents, query information, and interact with the model through a clean and 
intuitive interface.


#### ch01. 
1. docs
- https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/

2. commandlines
- embedding_responses meta
```
jq 'map(del(.data))' embedding_responses.collection.json
```

3. version 0.1.3
- stream response
- panel: rag-verbose, max_tokens, top_n, score_threshold
- supports epub, odt, odp, md
- cache embedding results by writting to json files
- show cost tokens for embedding
- show matching scores
- **logging**
- **run in container**
- **basic auth by using nginx**
