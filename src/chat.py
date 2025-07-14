#!/usr/bin/env python3
import os, sys, re, shutil
from pathlib import Path
# os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = "True"
sys.path.append(os.path.dirname(__file__))

import rag
from utils import now, file_md5

import litellm

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

def ui_update_str(sub, key):
    def fn(parameters, value):
        #print(f"<-- update {sub} {key}: {value}")
        parameters[sub][key] = value.strip()
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
    rag_prompt = f"{user_prompt}".format(input=user_input, context="\n\n".join(texts))

    return (rag_outputs, rag_prompt)


def call_llm(messages, parameters, stream=False):
    provider, model = parameters['llm']['selected_model'].split("/", 1)
    temperature = parameters['llm']['temperature']
    llm_models = parameters['llm_models']

    print("{} call_llm: provider={}, model={}, temperature={}, content={}".format(
        now(), provider, model, temperature,
        repr(messages[-1]['content']),
    ))

    found = next(
        (v for v in llm_models if v['provider'] == provider and v['model'] == model),
        None,
    )

    if found.get("hosted_vllm", False) is True:
        provider = "hosted_vllm"

    response = litellm.completion(
        custom_llm_provider=provider, model=model,
        api_base=found.get("api_base"), api_key=found.get("api_key"),
        max_tokens=parameters['llm']['max_tokens'], temperature=temperature,
        num_retries=3, timeout=60, stream=stream,
        messages=messages,
    )

    return response


def chat_fn(user_input, history, uploaded_files, parameters):
    # user_input = {"text": "hello", "files":["'/tmp/gradio/4d..."]} # when multimodal=True
    # print(f"<-- system_prompt: {system_prompt}")
    # print(f"<-- user_prompt: {user_prompt}")
    # print(f"<-- chat_fn: uploaded_files={uploaded_files}, parameters={parameters}")

    # TODO: how to add extract messages to history

    #### 1. init
    system_prompt = parameters['llm']['system_prompt'].strip()
    user_prompt = parameters['llm']['user_prompt'].strip()
    rag_enabled = parameters['rag']['enabled']
    stream = parameters['llm']['stream']

    user_input = user_input.strip()
    if user_input == "":
        return ({"role": "assitant", "content": "" }, history)

    #### 2. rag
    rag_outputs = []
    if rag_enabled and uploaded_files and len(user_prompt) > 0:
        rag_outputs, rag_prompt = handle_user_input(user_input, uploaded_files, parameters)
        if parameters['rag']['verbose']:
            user_input = rag_prompt

    #### 3. call llm
    messages = []
    if len(system_prompt) > 0:
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
    history.append(msg)
    # print(f"<-- messages: {messages}")

    #### 4. output
    # answer = user_input.upper() # Just a dummy response
    # reply = { "role": "assistant", "content": f"✨: {now()}, model={repr(model)}\n{answer}" }
    response = call_llm(messages, parameters, stream)

    if rag_enabled:
        msg['content'] = "{}: {}, temperature={}, rag_found={}\n{}".format(
            parameters['emoj']['rag'], now(),
            parameters['llm']['temperature'], len(rag_outputs),
            msg['content'],
        )
    else:
        msg['content'] = "{}: {}, temperature={}\n{}".format(
            parameters['emoj']['user'], now(),
            parameters['llm']['temperature'], msg['content'],
        )

    if stream:
        reply_content = "{}: {}, model={}\n".format(
            parameters['emoj']['ai'], now(), repr(parameters['llm']['selected_model'])
        )

        for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            reply_content += delta
            reply = {"role": "assistant", "content": reply_content }
            yield reply, history + [reply]
    else:
        answer = response.choices[0].message
        usage = response.usage

        print("<-- llm tokens usage: prompt={}, completion={}, total={}".format(
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens,
        ))

        reply_content = "{}: {}, model={}, pct_tokens=[{}, {}, {}]\n{}".format(
            parameters['emoj']['ai'], now(),
            repr(parameters['llm']['selected_model']), usage.prompt_tokens,
            usage.completion_tokens, usage.total_tokens,
            answer.content,
        )

        reply = { "role": answer.role, "content": reply_content }
        history.append(reply)
        # history.extend([msg, reply])
        #time.sleep(5)

        yield (reply, history) # Don't use return here
