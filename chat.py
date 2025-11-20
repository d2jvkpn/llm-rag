#!/usr/bin/env python3
import os, re, shutil
from pathlib import Path

import rag
from src.utils import now, file_md5


<<<<<<< HEAD
####
=======
>>>>>>> 9f8d5e8 (...)
def ui_update_fn(sub, key, min_val=None, max_val=None):
    def fn(parameters, value):
        if min_val is not None and value < min_val:
            value = min_val
<<<<<<< HEAD

        if max_val is not None and value > max_val:
            value = max_val

        print(f"<-- update {sub} {key}: {value}")
        parameters[sub][key] = value
        return parameters

    return fn

# deprecated
#def update_llm_textbox(parameters, key, value):
#    #print(f"<-- update_llm {key}: {value}")
#    config['llm'][key] = value
#    return value

####
def copy_gradio_files(paths, dirctory):
    docs = []
=======

        if max_val is not None and value > max_val:
            value = max_val

        print(f"<-- ui_update_fn {sub} {key}: {value}")
        parameters[sub][key] = value
        return parameters
>>>>>>> 9f8d5e8 (...)

    for p in paths:
        filename = Path(p).name
        doc_id = "md5-" + file_md5(p)
        #source_dir = os.path.dirname(p)
        #target_dir = os.path.join(dirctory, os.path.basename(source_dir))
        target_dir = Path(dirctory) / doc_id
        target_path = target_dir / filename
        # shutil.copy(p, save_path)
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


def copy_gradio_files(paths, dirctory):
    docs = []

    for p in paths:
        filename = Path(p).name
        doc_id = "md5-" + file_md5(p)
        #source_dir = os.path.dirname(p)
        #target_dir = os.path.join(dirctory, os.path.basename(source_dir))
        target_dir = Path(dirctory) / doc_id
        target_path = target_dir / filename
        # shutil.copy(p, save_path)
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


def handle_user_input(user_input, uploaded_files, parameters):
    if not parameters['rag']['enabled'] or not uploaded_files:
        return ([], user_input)

    paths = [v.name for v in uploaded_files]
    docs = copy_gradio_files(paths, parameters['http']['upload_dir'])
    # print(f"{now()} 📎 Uploaded: {docs}")
<<<<<<< HEAD
=======

    rag_outputs = rag.rag_query_docs(user_input, docs, parameters)
>>>>>>> 9f8d5e8 (...)

    rag_outputs = rag.rag_query_docs(user_input, docs, parameters)
    if len(rag_outputs) == 0:
        return ([], user_input)

    # print(f"<-- rag outputs: {rag_outputs}")
    texts = [f"#### {i+1}. {v}" for i, v in enumerate(rag_outputs)]

    user_prompt = parameters['llm']['user_prompt']
    user_input = f"{user_prompt}".format(input=user_input, context="\n\n".join(texts))

    return (rag_outputs, user_input)


def chat_fn(history, user_input, uploaded_files, system_prompt, user_prompt, parameters):
    # print(f"<-- system_prompt: {system_prompt}")
    # print(f"<-- user_prompt: {user_prompt}")
    # print(f"<-- parematers: selected_model={selected_model}, rag={rag}")

    # print(f"~~~ parameters: {parameters}")
    # TODO: how to add extract messages to history
    parameters['llm']['system_prompt'] = system_prompt
    parameters['llm']['user_prompt'] = user_prompt

    user_input = user_input.strip()
    if user_input == "":
        return (history, "")

    rag_outputs, user_input = handle_user_input(user_input, uploaded_files, parameters)

    messages = [{"role": "system", "content": parameters['llm']['system_prompt']}]

    for m in (history[-10:] if len(history) > 10 else history):
        # extract user_input only for rag message
        content = m['content'].split("\n", 1)[-1]

        if m['role'] == "user" and m['content'].startswith(parameters['emoj']['rag']):
            match = re.search(r"Input:\s*(.*?)\s*Context:", content, re.DOTALL)
            if match:
                content = match.group(1).strip()

        messages.append({"role": m['role'], "content": content})

    msg = { "role": "user", "content": user_input }
    messages.append(msg)
    # print(f"<-- messages: {messages}")

    # Just a dummy response
    # answer = user_input.upper()
    # reply = { "role": "assistant", "content": f"✨: {now()}, model={repr(model)}\n{answer}" }
    response = rag.call_llm(messages, parameters)
    ans = response.choices[0].message

    reply = {
        "role": ans.role,
        "content": "{}: {}, model={}, pct_tokens=[{}, {}, {}]\n{}".format(
            parameters['emoj']['ai'], now(), repr(parameters['llm']['selected_model']),
            response.usage.prompt_tokens, response.usage.completion_tokens,
            response.usage.total_tokens, ans.content,
        ),
    }

    if parameters['rag']['enabled']:
        msg['content'] = "{}: {}, temperature={}, matches={}\n{}".format(
            parameters['emoj']['ai'], now(),
            parameters['llm']['temperature'], len(rag_outputs), msg['content'],
        )
    else:
        msg['content'] = "{}: {}, temperature={}\n{}".format(
            parameters['emoj']['user'], now(),
            parameters['llm']['temperature'], msg['content'],
        )

    history.extend([msg, reply])
    #time.sleep(5)

    return (history, "")
