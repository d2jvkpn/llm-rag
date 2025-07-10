#!/usr/bin/env python3
import os, argparse, re, json, copy # time, shutil
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

parser.add_argument("--app", help="app name", default="rag-gadio")
parser.add_argument("--config", help="config path", default="./configs/local.yaml")

parser.add_argument("--reranker", help="enable reranker", action="store_true")
parser.add_argument("--delete-collection", help="delete collection in qdran", action="store_true")
parser.add_argument("--debug", help="debug mode", action="store_true")

parser.add_argument("--host", help="http listening host", default="127.0.0.1")
parser.add_argument("--port", help="http listening port", type=int, default=7860)
parser.add_argument("--share", help="gradio share", action="store_true")

args = parser.parse_args()
if args.debug:
    litellm._turn_on_debug()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)


##### dynamic parameters
config['llm']['temperature'] = 0.5
config['llm']['max_tokens'] = 1024
config['llm']['model_choices'] = [f"{v['provider']}/{v['model']}" for v in config['llm_models']]
config['llm']['selected_model'] = config['llm']['model_choices'][0]
config['llm']['system_prompt'] = config['llm']['system_prompt'].strip()
config['llm']['user_prompt'] = config['llm']['user_prompt'].strip()

config['rag'] = { "enabled": False }


#### static parameters
# emoj: 📚, 🤖, 🧑, 📝, 👀, ✨, 📄, 💬
config['app'] = args.app
config['upload_dir'] = os.path.join("data", "uploads") # args.app.replace(" ", "-")
config['reranker']['enabled'] = args.reranker

config['http']['share'] = args.share
config['http']['host'] = args.host
config['http']['port'] = args.port

_model = os.path.basename(config['embedding']['model']).replace(':', '--')
config['qdrant']['collection'] = f"{config['embedding']['provider']}__{_model}"

def static_parameters():
    d = config['embedding']
    embedding = { "provider": d['provider'], "model": d['model'] }

    d = config['qdrant']
    vector_db = { "collection": d['collection'], "top_n": d['top_n'] }

    d = config['reranker']
    reranker = { "enabled": d['enabled'], "top_n": d['top_n'] }

    #strs = [
    #    f"- embedding: {json.dumps(embedding)}",
    #    f"- vector_db: {json.dumps(vector_db)}",
    #    f"- reranker: {json.dumps(reranker)}",
    #]

    #return "Parameters\n" + "\n".join(strs)
    return "**Parameters**: " + \
        json.dumps({"embedding": embedding, "vector_db": vector_db, "reranker": reranker})


#### 2. setup
print(f"==> args: {args}")

os.makedirs(config['upload_dir'], exist_ok=True)
print(f"--> upload_dir: {config['upload_dir']}")


embed.init(config)

if args.delete_collection:
    collection = config['qdrant']['collection']

    if embed.QClient.collection_exists(collection):
        print(f"--> deleting collection: {collection=}")
        embed.QClient.delete_collection(collection)


if config['reranker']['enabled']:
    print("--> init_reranker:", config['reranker']['model'])
    local_llms.init_reranker(config['reranker']['model'])


#### 3. functions
def call_llm(messages, parameters):
    provider, model = parameters['llm']['selected_model'].split("/", 1)
    temperature = parameters['llm']['temperature']
    print(f"{now()} call_llm: provider={provider}, model={model}, temperature={temperature}")

    found = next(
        (v for v in config['llm_models'] if v['provider'] == provider and v['model'] == model),
        None,
    )

    if found.get("hosted_vllm", False) is True:
        provider = "hosted_vllm"

    response = litellm.completion(
        custom_llm_provider=provider, model=model,
        api_base=found.get("api_base"), api_key=found.get("api_key"),
        max_tokens=parameters['llm']['max_tokens'],
        temperature=temperature,
        num_retries=3, timeout=60, stream=False,
        messages=messages,
    )

    return response


def rag_query_docs(files, user_input):
    top_n = config['qdrant']['top_n']
    top_k = config['reranker']['top_n']

    if not files:
        return ([], [])

    paths = [v.name for v in files]
    docs = copy_gradio_files(paths, config['upload_dir'])
    # print(f"{now()} 📎 Uploaded: {docs}")
    for d in docs:
        embed.embedding_doc(d, process_doc.document2chunks)

    doc_ids = [d['doc_id'] for d in docs]

    vector = embed.litellm_embedding([user_input])[0]['data'][0]['embedding']
    # print(f"{now()} rag_query_docs vector: {vector}")
    hits = embed.search_doc(vector, doc_ids, top_n=top_n)
    print(f"{now()} search_doc: {len(hits.points)}")

    if len(hits.points) == 0:
        return (docs, [])

    if not config['reranker']['enabled'] or len(hits.points) <= top_k:
        return (docs, embed.points_to_chunks(hits.points, 64))

    #for p in hits.points:
    #    chunk_id = p.payload['chunk_id']
    #    page = p.payload['page']
    #    path = p.payload['path']
    #    print(f"<-- hits: chunk_id={chunk_id}, page={page}, path={path}")
    #    #print(f"    text: {p.payload['text']}")

    texts = [p.payload['text'] for p in hits.points]
    print(f"{now()} rerank_texts: {top_k}")
    scores = local_llms.rerank_texts(user_input, texts)
    points = [p for _, p in sorted(zip(scores, hits.points), reverse=True)][:top_k]

    return (docs, embed.points_to_chunks(points, 64))


def rag_user_input(parameters, files, user_input):
    if not parameters['rag']['enabled'] or not files:
        return ([], [], user_input)

    docs_files, rag_outputs = rag_query_docs(files, user_input)
    if len(rag_outputs) == 0:
        return (docs_files, [], user_input)

    # print(f"<-- rag outputs: {rag_outputs}")
    texts = [f"#### {i+1}. {v}" for i, v in enumerate(rag_outputs)]

    user_prompt = parameters['llm']['user_prompt']
    user_input = f"{user_prompt}".format(input=user_input, context="\n\n".join(texts))

    return (docs_files, rag_outputs, user_input)

#### 4. biz
def chat_func(history, user_input, files, system_prompt, user_prompt):
    # print(f"<-- system_prompt: {system_prompt}")
    # print(f"<-- user_prompt: {user_prompt}")
    # print(f"<-- parematers: selected_model={selected_model}, rag={rag}")

    # TODO: how to add extract messages to history
    parameters = {
        "llm": copy.deepcopy(config['llm']),
        "rag": copy.deepcopy(config['rag']),
    }
    parameters['llm']['system_prompt'] = system_prompt
    parameters['llm']['user_prompt'] = user_prompt

    user_input = user_input.strip()
    if user_input == "":
        return (history, "")

    docs_files, rag_outputs, user_input = rag_user_input(parameters, files, user_input)

    messages = [{"role": "system", "content": parameters['llm']['system_prompt']}]

    for m in (history[-5:] if len(history) > 5 else history):
        # extract user_input only for rag message
        content = m['content'].split("\n", 1)[-1]

        if m['role'] == "user" and m['content'].startswith("📝"):
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
    response = call_llm(messages, parameters)
    ans = response.choices[0].message

    reply = {
        "role": ans.role,
        "content": "✨: {}, model={}, pct_tokens=[{}, {}, {}]\n{}".format(
            now(), repr(parameters['llm']['selected_model']), 
            response.usage.prompt_tokens, response.usage.completion_tokens,
            response.usage.total_tokens, ans.content,
        ),
    }

    if parameters['rag']['enabled']:
        msg['content'] = "📝: {}, temperature={}, matches={}\n{}".format(
            now(), parameters['llm']['temperature'], len(rag_outputs), msg['content'],
        )
    else:
        msg['content'] = "📄: {}, temperature={}\n{}".format(
            now(), parameters['llm']['temperature'], msg['content'],
        )

    history.extend([msg, reply])
    #time.sleep(5)

    return (history, "")


def update_rag(key, value):
    print(f"<-- update_rag {key}: {value}")
    config['rag'][key] = value

def update_llm(key, value):
    print(f"<-- update_llm {key}: {value}")
    config['llm'][key] = value

def update_llm_min(key, value, min_val=None):
    if min_val and value < min_val:
        value = min_val

    print(f"<-- update_llm_min {key}: {value}")
    config['llm'][key] = value

# deprecated
def update_llm_textbox(key, value):
    #print(f"<-- update_llm {key}: {value}")
    config['llm'][key] = value
    return value

#### 5. run
with gr.Blocks(title=config['app']) as webui:
    upload_file_types = config['http']['upload_file_types']

    #param_info = {
    #    "temperature": {"type": "float", "description": "Creativity level", "default": 0.7},
    #    "max_tokens": {"type": "int", "description": "Max tokens in response", "default": 256}
    #}

    with gr.Row():
        with gr.Column(scale=3):
            system_prompt_input = gr.Textbox(
                interactive=True,
                label="System Prompt", lines=6, max_lines=6,
                value=config['llm']['system_prompt'],
            )

            with gr.Row():
                #clear_button = gr.Button("Clear", scale=1)
                with gr.Row():
                    temperature_slider = gr.Slider(
                        label="Temperature",
                        value=config['llm']['temperature'], minimum=0.0, maximum=1.5, step=0.1,
                    )

                    max_tokens_input = gr.Number(
                        label="max_tokens(min=20)",
                        value=config['llm']['max_tokens'],
                        precision=1,
                    )

                with gr.Row():
                    rag_checkbox = gr.Checkbox(label="RAG", value=config['rag']['enabled'])

            user_prompt_input = gr.Textbox(
                label="User Prompt for RAG, keep placeholder {input} and {context}",
                value=config['llm']['user_prompt'], lines=8, max_lines=8,
            )

            files_input = gr.File(
                label=f"Upload docs for RAG: {', '.join(upload_file_types)}",
                file_types=upload_file_types, file_count="multiple",
            )


        with gr.Column(scale=7):
            with gr.Row():
                gr.Markdown(static_parameters())
                #gr.ParamViewer(value=param_info)

            chatbot = gr.Chatbot(label="AI Assistant", type='messages', height=600)

            with gr.Row():
                with gr.Column(scale=9):
                    user_input = gr.Textbox(
                        show_label=False,
                        label="User input", lines=4, max_lines=4,
                        placeholder="Type your message here...",
                    )

                with gr.Column(scale=1, min_width=250):
                    send_button = gr.Button("Send", variant="secondary")

                    model_selector = gr.Dropdown(
                        show_label=False, interactive=True, label="Select Model",
                        value=config['llm']['selected_model'],
                        choices=config['llm']['model_choices'],
                    )

    rag_checkbox.change(
        fn=lambda value: update_rag("enabled", value),
        inputs=rag_checkbox, outputs=None,
    )

    #system_prompt_input.change(
    #    fn=lambda value: update_llm_textbox("system_prompt", value),
    #    inputs=system_prompt_input, outputs=system_prompt_input,
    #)

    max_tokens_input.change(
        fn=lambda value: update_llm_min("max_tokens", value, 20),
        inputs=max_tokens_input, outputs=None,
    )

    temperature_slider.change(
        fn=lambda value: update_llm("temperature", value),
        inputs=temperature_slider, outputs=None,
    )

    model_selector.change(
        fn=lambda value: update_llm("selected_model", value),
        inputs=model_selector, outputs=None,
    )

    ####
    inputs = [chatbot, user_input, files_input, system_prompt_input, user_prompt_input]

    # Submit message
    send_button.click(fn=chat_func, inputs=inputs, outputs=[chatbot, user_input])

    # Allow pressing enter
    user_input.submit(fn=chat_func, inputs=inputs, outputs=[chatbot, user_input])

    # Clear inputs
    #clear_button.click(
    #    fn=lambda: ("", "", None, []),
    #    inputs=[],
    #    outputs=[system_prompt_input, user_prompt_input, files_input, chatbot]
    #)

webui.launch(
    share=config['http']['share'],
    server_name=config['http']['host'],
    server_port=config['http']['port'],
)
