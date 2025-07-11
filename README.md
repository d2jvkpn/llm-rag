# RAG-Gradio
---
```meta
date: 2025-07-10
authors: []
version: 0.1.2
```


#### ch01. 
1. docs
- https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/

2. commandlines
- embedding_responses meta
```
jq 'map(del(.data))' embedding_responses.collection.json
```

3. next
- epub support
- logging
- show cost tokens for embedding
- show matching scores
- run in container
- basic auth by using nginx
