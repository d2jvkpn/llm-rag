#!/usr/bin/env python3
import os

import tiktoken


def doc_filename(path):
    filename = os.path.basename(path)

    if len(filename) > 64:
        filename = filename[:61] + "..."

    return filename


# gpt-3.5-turbo, gpt-4, text-davinci-003, cl100k_base
def count_tokens(text, model_name="gpt-4"):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))


def paragraphs_to_chunks(paragraphs, max_tokens):
    chunks, current, count = [], "", 0

    for p in paragraphs:
        count += count_tokens(p)
        current += p + "\n"

        if count >= max_tokens:
            chunks.append(current.strip())
            current, count = "", 0

    if current:
        chunks.append(current.strip())

    return chunks
