#!/usr/bin/env python3
import os, argparse, re, json # time, shutil
os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = "True"

from src import process_doc, embed, local_llms
from src.utils import now, copy_gradio_files

import yaml, litellm
import gradio as gr

#py = os.path.abspath(os.sys.argv[0])
#app = os.path.basename(os.path.dirname(py))


#### 1. configuration
parser = argparse.ArgumentParser(
    description="parse commandline arguments",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--config", help="config path", default="./configs/local.yaml")
parser.add_argument("--max_tokens", help="max tokens", type=int, default=1024)
parser.add_argument("--reranker", help="enable reranker", action="store_true")
parser.add_argument("--delete-collection", help="delete collection in qdran", action="store_true")
parser.add_argument("--host", help="http listening host", default="127.0.0.1")
parser.add_argument("--port", help="http listening port", type=int, default=7860)
parser.add_argument("--share", help="gradio share", action="store_true")
parser.add_argument("--debug", help="debug mode", action="store_true")

args = parser.parse_args()
if args.debug:
    litellm._turn_on_debug()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

config['upload_dir'] = os.path.join("data", "uploads") # args.app.replace(" ", "-")
config['reranker']['enabled'] = args.reranker

config['llm']['max_tokens'] = args.max_tokens
config['llm']['system_prompt'] = config['llm']['system_prompt'].strip()
config['llm']['user_prompt'] = config['llm']['user_prompt'].strip()

config['http']['share'] = args.share
config['http']['host'] = args.host
config['http']['port'] = args.port

embedding_provider = config['embedding']['provider']
embedding_model = os.path.basename(config['embedding']['model'])
config['qdrant']['collection'] = f"{embedding_provider}__{embedding_model.replace(':', '--')}"

def get_parameters():
    d = config['llm']
    _llm = { "max_tokens": d['max_tokens'] }

    _embedding = { "provider": embedding_provider, "model": embedding_model }

    d = config['qdrant']
    _vector_db = { "collection": d['collection'], "top_n": d['top_n'] }

    d = config['reranker']
    _reranker = { "enabled": d['enabled'], "top_n": d['top_n'] }

    strs = [
        f"- llm: {json.dumps(_llm)}",
        f"- embedding: {json.dumps(_embedding)}",
        f"- vector_db: {json.dumps(_vector_db)}",
        f"- reranker: {json.dumps(_reranker)}",
    ]

    return "\n".join(strs)


#### 2. setup
print(f"==> args: {args}")

os.makedirs(config['upload_dir'], exist_ok=True)
print(f"--> upload_dir: {config['upload_dir']}")


embed.init(config)
print(f"--> using collection: {config['qdrant']['collection']}")

if args.delete_collection:
    collection = config['qdrant']['collection']

    if embed.QClient.collection_exists(collection):
        print(f"--> deleting collection: {collection=}")
        embed.QClient.delete_collection(collection)


if config['reranker']['enabled']:
    print("--> init_reranker:", config['reranker']['model'])
    local_llms.init_reranker(config['reranker']['model'])


#### 3. functions
def call_llm(messages, selected_model, temperature):
    provider, model = selected_model.split("/", 1)
    print(f"{now()} --> call_llm: provider={provider}, model={model}, temperature={temperature}")

    found = next(
        (v for v in config['llm_models'] if v['provider'] == provider and v['model'] == model),
        None,
    )

    if found.get("hosted_vllm", False) is True:
        provider = "hosted_vllm"

    response = litellm.completion(
        custom_llm_provider=provider, model=model,
        api_base=found.get("api_base"), api_key=found.get("api_key"),
        max_tokens=config['llm']['max_tokens'],
        temperature=temperature,
        num_retries=3, timeout=60, stream=False,
        messages=messages,
    )

    return response


def rag_docs(files, user_input):
    top_n = config['qdrant']['top_n']
    top_k = config['reranker']['top_n']

    if not files:
        return ([], [])

    paths = [v.name for v in files]
    docs = copy_gradio_files(paths, config['upload_dir'])
    # print(f"{now()} --> 📎 Uploaded: {docs}")
    embed.embedding_docs(docs, process_doc.document2chunks)
    doc_ids = [d['doc_id'] for d in docs]

    vector = embed.litellm_embedding([user_input])[0]
    # print(f"{now()} --> rag_docs vector: {vector}")
    hits = embed.search_doc(vector, doc_ids, top_n=top_n)
    print(f"{now()} --> retrieved chunks: {len(hits.points)}")

    if len(hits.points) == 0:
        return (docs, [])

    if not config['reranker']['enabled'] or len(hits.points) <= top_k:
        return (docs, embed.points_to_chunks(hits.points, 64))

    #for p in hits.points:
    #    chunk_id = p.payload['chunk_id']
    #    page = p.payload['page']
    #    path = p.payload['path']
    #    print(f"--> hits: chunk_id={chunk_id}, page={page}, path={path}")
    #    #print(f"    text: {p.payload['text']}")

    texts = [p.payload['text'] for p in hits.points]
    print(f"{now()} --> rerank_texts: {top_k}")
    scores = local_llms.rerank_texts(user_input, texts)
    points = [p for _, p in sorted(zip(scores, hits.points), reverse=True)][:top_k]

    return (docs, embed.points_to_chunks(points, 64))


#### 4. biz
def chat_func(history, user_input, files,
    system_prompt, user_prompt, selected_model, rag, temperature):
    # print(f"--> system_prompt: {system_prompt}")
    # print(f"--> user_prompt: {user_prompt}")
    # print(f"--> parematers: selected_model={selected_model}, rag={rag}")

    # TODO: how to add extract messages to history

    user_input = user_input.strip()
    if user_input == "":
        return (history, "")

    docs = []
    if rag:
        docs, outputs = rag_docs(files, user_input)
        if len(outputs) > 0:
            # print(f"--> rag outputs: {outputs}")
            texts = [f"#### {i+1}. {v}" for i, v in enumerate(outputs)]
            context = "\n\n".join(texts)
            user_input = f"{user_prompt}".format(input=user_input, context=context)

    messages = [{"role": "system", "content": system_prompt}]

    for m in (history[-5:] if len(history) > 5 else history):
        # extract user_input only for rag message
        content = m['content'].split("\n", 1)[-1]

        if m['role'] == "user" and m['content'].startswith("📚"):
            match = re.search(r"Input:\s*(.*?)\s*Context:", content, re.DOTALL)
            if match:
                content = match.group(1).strip()

        messages.append({"role": m['role'], "content": content})

    msg = { "role": "user", "content": user_input }
    messages.append(msg)
    # print(f"--> messages: {messages}")

    # Just a dummy response
    # answer = user_input.upper()
    # reply = { "role": "assistant", "content": f"🤖: {now()}, model={repr(model)}\n{answer}" }
    response = call_llm(messages, selected_model, temperature)
    message = response.choices[0].message
    usage = response.usage
    usage = [usage.prompt_tokens, usage.completion_tokens, usage.total_tokens]
    tokens = f"prompt={usage[0]}, completion={usage[1]}, total={usage[2]}"
    print(f"{now()} --> llm_tokens: {tokens}")

    reply = {
        "role": message.role,
        "content": f"🤖: {now()}, model={repr(selected_model)}, {tokens}\n{message.content}",
    }

    if rag:
        msg['content'] = f"📚: {now()}, temperature={temperature}\n{msg['content']}"
    else:
        msg['content'] = f"🧑: {now()}, temperature={temperature}\n{msg['content']}"

    history.extend([msg, reply])
    #time.sleep(5)

    return (history, "")


with gr.Blocks(title=os.getenv("app", "rag-gradio")) as webui:
    upload_file_types = config['http']['upload_file_types']
    model_choices = [f"{v['provider']}/{v['model']}" for v in config['llm_models']]

    #param_info = {
    #    "temperature": {"type": "float", "description": "Creativity level", "default": 0.7},
    #    "max_tokens": {"type": "int", "description": "Max tokens in response", "default": 256}
    #}

    with gr.Row():
        with gr.Column(scale=3):
            system_prompt_input = gr.Textbox(
                interactive=True,
                label="System Prompt", lines=7, max_lines=7,
                value=config['llm']['system_prompt'],
            )

            with gr.Row():
                gr.Markdown(f"Parameters\n{get_parameters()}")
                #gr.ParamViewer(value=param_info)

            with gr.Row():
                temperature = gr.Slider(
                    label="Temperature",
                    value=0.5, minimum=0.0, maximum=1.5, step=0.1,
                )

                rag = gr.Checkbox(label="Enable RAG", value=False)
                #clear_button = gr.Button("Clear", scale=1)

            user_prompt_input = gr.Textbox(
                label="User Prompt for RAG, keep placeholder {input} and {context}",
                value=config['llm']['user_prompt'], lines=10, max_lines=10,
            )

            files_input = gr.File(
                label=f"Upload docs for RAG: {', '.join(upload_file_types)}",
                file_types=upload_file_types, file_count="multiple",
            )


        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="AI Assistant", type='messages', height=1000)

            with gr.Row():
                with gr.Column(scale=9):
                    user_input = gr.Textbox(
                        show_label=False,
                        label="User input", lines=4, max_lines=4,
                        placeholder="Type your message here...",
                    )

                with gr.Column(scale=1, min_width=250):
                    send_button = gr.Button("Send", variant="primary")

                    model_selector = gr.Dropdown(
                        show_label=False, interactive=True, label="Select Model",
                        value=model_choices[0], choices=model_choices,
                    )

    inputs=[
        chatbot, user_input, files_input, system_prompt_input, user_prompt_input,
        model_selector, rag, temperature,
    ]

    outputs=[chatbot, user_input]

    send_button.click(fn=chat_func, inputs=inputs, outputs=outputs) # Submit message
    user_input.submit(fn=chat_func, inputs=inputs, outputs=outputs) # Allow pressing enter

    # Clear inputs
    #clear_button.click(
    #    fn=lambda: ("", "", None, []),
    #    inputs=[],
    #    outputs=[system_prompt_input, user_prompt_input, files_input, chatbot]
    #)


#### 5. run
webui.launch(
    share=config['http']['share'],
    server_name=config['http']['host'],
    server_port=config['http']['port'],
)
