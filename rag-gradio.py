#!/usr/bin/env python3
import os, argparse, time, re, shutil
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from src import process_doc, embed
from src.utils import now, file_md5, move_gradio_files

import yaml, litellm
import gradio as gr

#py = os.path.abspath(os.sys.argv[0])
#app = os.path.basename(os.path.dirname(py))

parser = argparse.ArgumentParser(
    description="parse commandline arguments",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--app", help="app name", default="ai-rag-a02")
parser.add_argument("--config", help="config path", default="./configs/local.yaml")
parser.add_argument("--max_tokens", help="max tokens", type=int, default=1024)
parser.add_argument("--host", help="http listening host", default="127.0.0.1")
parser.add_argument("--port", help="http listening port", type=int, default=7860)
parser.add_argument("--share", help="gradio share", action="store_true")
parser.add_argument("--debug", help="debug mode", action="store_true")
parser.add_argument("--delete-collection", help="delete collection in qdran", action="store_true")

args = parser.parse_args()
if args.debug:
    litellm._turn_on_debug()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

config["system_prompt"] = config["system_prompt"].strip()
config["user_prompt"] = config["user_prompt"].strip()

UPLOAD_DIR = os.path.join("data", args.app.replace(" ", "-"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

embed.init(args.config)
if args.delete_collection:
    collection = config["qdrant"]["collection"]

    if embed.QClient.collection_exists(collection):
        print(f"--> deleting collection: {collection=}")
        embed.QClient.delete_collection(collection)


def call_llm(messages, selected_model, temperature):
    provider, model = selected_model.split("/", 1)
    print(f"{now()} --> call_llm: provider={provider}, model={model}, temperature={temperature}")

    found = next(
        (v for v in config["llm_models"] if v["provider"] == provider and v["model"] == model),
        None,
    )

    if found.get("hosted_vllm", False) is True:
        provider = "hosted_vllm"

    response = litellm.completion(
        custom_llm_provider=provider, model=model,
        api_base=found.get("api_base"), api_key=found.get("api_key"),
        max_tokens=args.max_tokens,
        temperature=temperature,
        num_retries=3, timeout=60, stream=False,
        messages=messages,
    )

    msg = response.choices[0].message
    return msg


def embedding_docs(docs):
    for d in docs:
        doc_id = "md5-" + d["md5"]
        doc_path = repr(d["path"])

        if embed.vectordb_doc_exists(doc_id):
            print(f"{now()} ---> embedding_docs skip: {doc_path}")
            continue

        print(f"{now()} ---> document2chunks: {d}")
        doc = process_doc.document2chunks(d["path"], doc_id)

        texts = [c["text"]for c in doc["chunks"]]
        print(f"{now()} ---> litellm_embedding: chunks={len(texts)}, doc_path={doc_path}")
        vectors = embed.litellm_embedding(texts)

        print(f"{now()} ---> vectordb_save: chunks={len(texts)}, doc_path={doc_path}")
        embed.vectordb_save(doc, vectors, recreate=False)


def rag_docs(files, user_input):
    doc_ids = []
    top_n = config["qdrant"]["top_n"]
    top_k = config["rerank"]["top_n"]

    if files:
        #file_names = [os.path.basename(v.name) for v in files]
        docs = move_gradio_files(files, UPLOAD_DIR)
        # print(f"{now()} --> 📎 Uploaded: {docs}")
        embedding_docs(docs)
        doc_ids = ["md5-" + d["md5"] for d in docs]

    if len(doc_ids) == 0:
        return ""

    vector = embed.litellm_embedding(user_input)
    # print(f"{now()} --> rag_docs vector: {vector}")
    hits = embed.search_doc(vector[0], doc_ids, top_n=top_n)
    print(f"{now()} --> retrieved chunks: {len(hits.points)}")

    if args.debug:
        print(f"==> Hits:\n{hits}")

    if len(hits.points) == 0:
        return ""

    texts = [f"- {p.payload['text']}" for p in hits.points]
    if len(hits.points) <= top_k + 3:
        return "\n\n".join(texts)

    # TODO: reranker
    return "\n\n".join(texts)

def chat_func(history, user_input, files,
    system_prompt, user_prompt, selected_model, rag, temperature):
    # print(f"--> system_prompt: {system_prompt}")
    # print(f"--> user_prompt: {user_prompt}")
    # print(f"--> parematers: selected_model={selected_model}, rag={rag}")

    user_input = user_input.strip()
    if user_input == "":
        return [history, ""]

    if rag:
        context = rag_docs(files, user_input)
        if context:
            user_input = f"{user_prompt}".format(input=user_input, context=context)
        else:
            print("{} --> rag not context")

    messages = [{"role": "system", "content": system_prompt}]

    for m in (history[-5:] if len(history) > 5 else history):
        # extract user_input only for rag message
        content = m["content"].split("\n", 1)[-1]

        if m["role"] == "user" and m["content"].startswith("📚"):
            match = re.search(r"Input:\s*(.*?)\s*Context:", content, re.DOTALL)
            if match:
                content = match.group(1).strip()

        messages.append({"role": m["role"], "content": content})

    msg = { "role": "user", "content": user_input }
    messages.append(msg)
    # print(f"--> messages: {messages}")

    # Just a dummy response
    # answer = user_input.upper()
    # reply = { "role": "assistant", "content": f"🤖: {now()}, model={repr(model)}\n{answer}" }

    reply = call_llm(messages, selected_model, temperature)
    reply = {
        "role": reply.role,
        "content": f"🤖: {now()}, selected_model={repr(selected_model)}\n{reply.content}",
    }

    if rag:
        msg["content"] = f"📚: {now()}, temperature={temperature}\n{msg['content']}"
    else:
        msg["content"] = f"🧑: {now()}, temperature={temperature}\n{msg['content']}"

    history.extend([msg, reply])
    #time.sleep(5)

    return [history, ""]


with gr.Blocks(title=args.app) as view:
    upload_file_types = config['http']['upload_file_types']
    model_choices = [f"{v['provider']}/{v['model']}" for v in config["llm_models"]]

    with gr.Row():
        with gr.Column(scale=1):
            system_prompt_input = gr.Textbox(
                label="System Prompt", value=config["system_prompt"],
                lines=8, max_lines=8, interactive=True,
            )

        with gr.Column(scale=1):
            user_prompt_input = gr.Textbox(
                label="User Prompt for RAG, keep placeholder {input} and {context}",
                value=config["user_prompt"],
                lines=8, max_lines=8,
            )

        with gr.Column(scale=1):
            files_input = gr.File(
                label=f"Upload docs for RAG: {', '.join(upload_file_types)}",
                file_types=upload_file_types, file_count="multiple",
            )

    chatbot = gr.Chatbot(label="AI Assistant", type='messages', height=500)

    with gr.Row():
        with gr.Column(scale=12):
            user_input = gr.Textbox(
                show_label=False, label="User input", placeholder="Type your message here...",
                lines=4, max_lines=4,
            )

        with gr.Column(scale=1, min_width=250):
            temperature = gr.Slider(
                label="Temperature",
                value=0.7, minimum=0.0, maximum=1.0, step=0.1,
            )

            rag = gr.Checkbox(label="Enable RAG", value=False)
            #clear_button = gr.Button("Clear", scale=1)


        with gr.Column(scale=1, min_width=250):
            send_button = gr.Button("Send", variant="primary", scale=1)

            model_selector = gr.Dropdown(
                show_label=False, interactive=True, label="Select Model",
                value=model_choices[0], choices=model_choices,
            )

    inputs=[
        chatbot, user_input, files_input, system_prompt_input, user_prompt_input,
        model_selector, rag, temperature,
    ]

    outputs=[chatbot, user_input]

    # Submit message
    send_button.click(fn=chat_func, inputs=inputs, outputs=outputs)

    # Allow pressing enter
    user_input.submit(fn=chat_func, inputs=inputs, outputs=outputs)

    # Clear inputs
    #clear_button.click(
    #    fn=lambda: ("", "", None, []),
    #    inputs=[],
    #    outputs=[system_prompt_input, user_prompt_input, files_input, chatbot]
    #)

view.launch(share=args.share, server_name=args.host, server_port=args.port)
