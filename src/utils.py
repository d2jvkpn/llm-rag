#!/usr/bin/env python3
import os, hashlib, shutil
from datetime import datetime


def now():
    at = datetime.now().astimezone()
    # return f"{at}".replace(" ", "T")
    return at.strftime("%Y-%m-%dT%H:%M:%S%z")


def file_md5(file_path):
    #with open(filepath, 'rb') as f:
    #    content = f.read()
    #return hashlib.md5(content).hexdigest()

    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def copy_gradio_files(paths, dirctory):
    docs = []

    for p in paths:
        filename = os.path.basename(p)
        doc_id = "md5-" + file_md5(p)
        #source_dir = os.path.dirname(p)
        #target_dir = os.path.join(dirctory, os.path.basename(source_dir))
        target_dir = os.path.join(dirctory, doc_id)
        target_path = os.path.join(target_dir, filename)
        # shutil.copy(p, save_path)
        doc = { "path": target_path, "doc_id": doc_id, "exists": False }

        if os.path.exists(target_path) and os.path.isfile(target_path):
            doc["exists"] = True
        else:
            os.makedirs(target_dir, exist_ok=True)
            print(f"{now()} --> copy_gradio_files: {p} -> {target_path}")
            shutil.copy(p, target_path)

        #if os.path.isdir(source_dir):
        #    print(f"--> remove duplicated: {p}")
        #    shutil.rmtree(source_dir)

        docs.append(doc)

    return docs
