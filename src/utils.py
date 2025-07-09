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


def move_gradio_files(files, dirctory):
    docs = []

    for file in files:
        print(f"{now()} --> process_uploaded_files: {file}")
        filename = os.path.basename(file.name)
        source_dir = os.path.dirname(file.name)
        md5 = file_md5(file.name)
        #target_dir = os.path.join(dirctory, os.path.basename(file_dir))
        target_dir = os.path.join(dirctory, "md5-" + md5)
        target_path = os.path.join(target_dir, filename)
        # shutil.copy(file.name, save_path)
        doc = { "path": target_path, "md5": md5, "exists": True}

        if os.path.exists(target_path) and os.path.isfile(target_path):
            doc["exists"] = False
        else:
            os.makedirs(target_dir, exist_ok=True)
            print(f"--> copy file: {file.name} -> {target_path}")
            shutil.copy(file.name, target_path)

        #if os.path.isdir(source_dir):
        #    print(f"--> remove duplicated: {file.name}")
        #    shutil.rmtree(source_dir)

        docs.append(doc)

    return docs
