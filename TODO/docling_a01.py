#!/usr/bin/env python3
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker, HierarchicalChunker


# 1. 转换文档为 DoclingDocument
converter = DocumentConverter()
result = converter.convert(source="your_file.pdf")
dl_doc = result.document


# 2. 使用 HierarchicalChunker 或 HybridChunker
# HierarchicalChunker：基于文档结构拆分，每个元素按自然结构 chunk
hier_chunker = HierarchicalChunker()
hier_chunks = list(hier_chunker.chunk(dl_doc))

# HybridChunker：在 hierarchical 基础上进一步按 token limit 拆分或合并
hybrid_chunker = HybridChunker()
hybrid_chunks = list(hybrid_chunker.chunk(dl_doc))


# 3. 打印示例 chunk 内容与 metadata
for i, chunk in enumerate(hybrid_chunks[:5]):
    print(f"--- chunk {i} ---")
    print("text:", chunk.text[:200], "…")
    print("metadata:", chunk.metadata)
    print("contextualized:", hybrid_chunker.contextualize(chunk)[:200], "…\n")
