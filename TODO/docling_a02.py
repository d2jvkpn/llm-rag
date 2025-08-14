#!/usr/bin/env python3
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain.embeddings import HuggingFaceEmbeddings
from docling.chunking import HybridChunker

loader = DoclingLoader(
    file_path=["your_doc.pdf"],
    export_type=ExportType.DOC_CHUNKS,               # 默认模式：自动 chunk
    chunker=HybridChunker(tokenizer=EMBED_MODEL_ID)  # 指定 HybridChunker
)

docs = loader.load()  # 内部使用的是 HybridChunker

# from langchain.vectorstores import Milvus
# 然后生成 embedding、存入 Milvus、构建检索等
