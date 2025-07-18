#!/usr/bin/env python3
import os, json, shutil
from pathlib import Path
from datetime import datetime

from .utils import now, file_md5

import gradio as gr


def update_key(key):
    def fn(parameters, value):
        #print(f"<-- update {key}: {value}")
        if type(value) == str:
            value = value.strip()

        parameters[key] = value
        return

    return fn


def update_sub_key_minmax(sub, key, min_val=None, max_val=None):
    def fn(parameters, value):
        if min_val is not None and value < min_val:
            value = min_val

        if max_val is not None and value > max_val:
            value = max_val

        #print(f"{now()} <-- update {sub} {key}: {value}")
        parameters[sub][key] = value
        return

    return fn

def update_sub_key(sub, key):
    def fn(parameters, value):
        #print(f"<-- update {sub} {key}: {value}")
        if type(value) == str:
            value = value.strip()

        parameters[sub][key] = value
        return

    return fn

# deprecated
#def update_llm_textbox(parameters, key, value):
#    #print(f"<-- update_llm {key}: {value}")
#    config['llm'][key] = value
#    return value

def export_chat(history):
    if not history or len(history) == 0:
        return "Export chat to JSON 📄"
        # raise ValueError("no history")

    filename = f"chat_{datetime.now().strftime('%Y-%m-%d-%s')}.json"
    directory = Path("data") / "chat"
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / filename

    with open(str(filepath), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return f"Saved to {filename}"


def copy_gradio_files(paths, dirctory):
    docs = []

    for p in paths:
        filename = Path(p).name
        doc_id = "md5-" + file_md5(p)
        #source_dir = os.path.dirname(p)
        #target_dir = os.path.join(dirctory, os.path.basename(source_dir))

        target_dir = Path(dirctory) / doc_id
        target_path = target_dir / filename

        doc = { "path": target_path, "doc_id": doc_id, "exists": False }

        if target_path.exists() and target_path.is_file():
            doc['exists'] = True
        else:
            os.makedirs(target_dir, exist_ok=True)
            print(f"{now()} copy_gradio_files: {p} -> {target_path}")
            shutil.copy(p, target_path)

        #if os.path.isdir(source_dir):
        #    print(f"{now()} remove duplicated: {p}")
        #    shutil.rmtree(source_dir)

        docs.append(doc)

    return docs


class JSONLogger(gr.FlaggingCallback):
    def __init__(self, keys, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.keys = keys

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.log_dir / f"flagging.{datetime.now().strftime('%F')}.jsonl"
        self.file = None

    def setup(self, components, flagging_options):
        #print("~~~ Flagging setup")
        if not self.filepath.exists():
            self.file = open(self.filepath, "w", encoding="utf-8")
        else:
            self.file = open(self.filepath, "a", encoding="utf-8")

    def flag(self, flag_data, flag_option=None, username=None):
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%:z")
        record = {"timestamp": timestamp}

        if flag_option is not None:
            record["flag"] = flag_option.lower()

        for i, k in enumerate(self.keys):
            record[k] = flag_data[i]

        json.dump(record, self.file, ensure_ascii=False)
        self.file.write("\n")
        self.file.flush()

    def close(self):
        #print("~~~ Closing flagging callback and cleaning up")
        if self.file is not None:
            self.file.close()
