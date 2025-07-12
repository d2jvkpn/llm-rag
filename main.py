#!/usr/bin/env python3
import os, argparse, json, copy # time, shutil
from pathlib import Path
os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = "True"

from src import embed, local_llms
from src.utils import now, get_local_ip
from chat import chat_fn, ui_update_fn, ui_update_str

import yaml, uuid # litellm
import gradio as gr


#### 1. configuration
parser = argparse.ArgumentParser(
    description="parse commandline arguments",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--app", help="app name", default="RAG-Gradio")
parser.add_argument("--config", help="config path", default="./configs/local.yaml")

parser.add_argument("--reranker", help="enable reranker", action="store_true")

parser.add_argument(
    "--delete-collection",
    help="delete collection in vector database",
    action="store_true",
)

parser.add_argument("--host", help="http listening host", default="127.0.0.1")
parser.add_argument("--port", help="http listening port", type=int, default=7861)
parser.add_argument("--share", help="gradio share", action="store_true")
#parser.add_argument("--debug", help="debug mode", action="store_true")
parser.add_argument("--mode", help="running mode", default="dev")

args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)


##### dynamic parameters
config['llm']['temperature'] = 0.7
config['llm']['max_tokens'] = 1000
_model_choices = [f"{v['provider']}/{v['model']}" for v in config['llm_models']]
config['llm']['model_choices'] = _model_choices
config['llm']['selected_model'] = _model_choices[0]
config['llm']['system_prompt'] = config['llm']['system_prompt'].strip()
config['llm']['user_prompt'] = config['llm']['user_prompt'].strip()

config['rag'] = { "enabled": False }


#### static parameters
config['app'] = { "name": args.app, "mode": args.mode }

# emoj: 📚, 🤖, 🧑, 📝, 👀, ✨, 📄, 💬, 🔍, 🐦‍⬛, 🦉, 🪶
config['emoj'] = {
  "user": "📝",
  "ai": "✨",
  "rag": "📚",
}

# args.app.replace(" ", "-")
config['http']['upload_dir'] = Path("data") / "uploads"
config['http']['share'] = args.share
config['http']['host'], config['http']['port'] = args.host, args.port

config['reranker']['enabled'] = args.reranker

css = Path("assets") / "style.css"
if css.exists():
    with open(css, 'r') as f:
        config['http']['css'] = f.read()

js = Path("assets") / "gradio.js"
if js.exists():
    with open(js, 'r') as f:
        config['http']['js'] = f.read()

_model = Path(config['embedding']['model']).name.replace(':', '--')
config['qdrant']['collection'] = f"{config['embedding']['provider']}__{_model}"


#### 2. setup
print(f"{now()} ==> args: {args}")

os.makedirs(config['http']['upload_dir'], exist_ok=True)
# print(f"--> upload_dir: {config['http']['upload_dir']}")

embed.init(config)

if args.delete_collection:
    collection = config['qdrant']['collection']

    if embed.QClient.collection_exists(collection):
        print(f"--> deleting collection: {collection}")
        embed.QClient.delete_collection(collection)

if config['reranker']['enabled']:
    print(f"{now()} init_reranker:", config['reranker']['model'])
    local_llms.init_reranker(config['reranker']['model'])
    print(f"{now()} reranker initialized")


#### 3. functions
def display_parameters(mode):
    #strs = [
    #    f"- embedding: {json.dumps(embedding)}",
    #    f"- vector_db: {json.dumps(vector_db)}",
    #    f"- reranker: {json.dumps(reranker)}",
    #]

    d = config['embedding']
    embedding = { "provider": d['provider'], "model": d['model'] }

    d = config['qdrant']
    vector_db = { "collection": d['collection'] }

    d = config['reranker']
    reranker = { "model": os.path.basename(d['model']), "enabled": d['enabled'] }

    parameters = {"reranker": reranker }
    if mode == "dev":
        parameters = {"reranker": reranker, "embedding": embedding, "vector_db": vector_db }

    #return "Parameters\n" + "\n".join(strs)
    return "**Parameters**: " + json.dumps(parameters)


def panel_visibility(current_state):
    new_state = not current_state
    return gr.update(visible=new_state), new_state

#### 5. run
with gr.Blocks(
    title=config['app']['name'], css=config['http'].get('css'),
    js=config['http'].get('js'),
) as webui:
    #param_info = {
    #    "temperature": {"type": "float", "description": "Creativity level", "default": 0.7},
    #    "max_tokens": {"type": "int", "description": "Max tokens in response", "default": 256}
    #}

    upload_file_types = config['http']['upload_file_types']

    parameters = gr.State({
        "emoj": copy.deepcopy(config['emoj']),
        "http": copy.deepcopy(config['http']),
        "reranker": copy.deepcopy(config['reranker']),
        "llm_models": copy.deepcopy(config['llm_models']),

        "account": { },
        "qdrant": copy.deepcopy(config['qdrant']),
        "llm": copy.deepcopy(config['llm']),
        "rag": copy.deepcopy(config['rag']),
    })

    panel_state = gr.State(False)

    with gr.Row():
        with gr.Column(scale=3, elem_classes=["my-column"], visible=True) as panel_column:
            # gr.HTML('<h4 style="margin: 0"> Control panel </h4>')
            system_prompt_input = gr.Textbox(
                interactive=True, label="System Prompt", lines=6, max_lines=6,
                value=config['llm']['system_prompt'],
            )

            user_prompt_input = gr.Textbox(
                label="User Prompt for RAG, keep placeholder {input} and {context}",
                lines=10, max_lines=10,
                value=config['llm']['user_prompt'],
            )

            with gr.Row():
                #clear_button = gr.Button("Clear", scale=1)
                with gr.Row():
                    max_tokens_input = gr.Number(
                        label="max_tokens(min=20)",
                        value=config['llm']['max_tokens'], precision=1,
                    )

                    temperature_slider = gr.Slider(
                        label="temperature",
                        value=config['llm']['temperature'], minimum=0.0, maximum=1.5, step=0.1,
                    )

                with gr.Row():
                    rag_checkbox = gr.Checkbox(label="RAG", value=config['rag']['enabled'])

                    top_n_slider = gr.Slider(
                        label="top n",
                        value=config['qdrant']['top_n'], minimum=1, maximum=30, step=1,
                    )

                    score_threshold_slider = gr.Slider(
                        label="score threshold",
                        value=config['qdrant']['score_threshold'],
                        minimum=0.1, maximum=1.0, step=0.01,
                    )


            uploaded_files = gr.File(
                label=f"Upload docs for RAG: {', '.join(upload_file_types)}",
                file_types=upload_file_types, file_count="multiple",
            )


        with gr.Column(scale=7, elem_classes=["my-column"]):
            with gr.Row():
                toggle_panel = gr.Button("Panel", elem_id="toggle-panel")

                model_selector = gr.Dropdown(
                    interactive=True, show_label=False, label="Select Model",
                    value=config['llm']['selected_model'],
                    choices=config['llm']['model_choices'],
                    elem_id="model-selector", scale=2,
                )

                with gr.Column(scale=8):
                    gr.Markdown(display_parameters(config['app']['mode']))
                #gr.ParamViewer(value=param_info)

            # height=600
            chatbot = gr.Chatbot(label="AI Assistant", type='messages', elem_id="my-chatbot")

            gr.ChatInterface(
                chat_fn, type="messages", chatbot=chatbot, multimodal=False,
                additional_inputs=[uploaded_files, parameters], additional_outputs=[chatbot],
                autofocus=True,
            )

    #system_prompt_input.change(
    #    fn=lambda value: update_llm_textbox("system_prompt", value),
    #    inputs=system_prompt_input, outputs=system_prompt_input,
    #)

    ####
    system_prompt_input.change(
        fn=ui_update_str("llm", "system_prompt"),
        inputs=[parameters, system_prompt_input], outputs=[parameters],
    )

    user_prompt_input.change(
        fn=ui_update_str("llm", "user_prompt"),
        inputs=[parameters, user_prompt_input], outputs=[parameters],
    )

    ####
    max_tokens_input.change(
        fn=ui_update_fn("llm", "max_tokens", 20),
        inputs=[parameters, max_tokens_input], outputs=[parameters],
    )

    temperature_slider.change(
        fn=ui_update_fn("llm", "temperature"),
        inputs=[parameters, temperature_slider], outputs=[parameters],
    )

    ####
    rag_checkbox.change(
        fn=ui_update_fn("rag", "enabled"),
        inputs=[parameters, rag_checkbox], outputs=[parameters],
    )

    top_n_slider.change(
        fn=ui_update_fn("qdrant", "top_n"),
        inputs=[parameters, top_n_slider], outputs=[parameters],
    )

    score_threshold_slider.change(
        fn=ui_update_fn("qdrant", "score_threshold"),
        inputs=[parameters, score_threshold_slider], outputs=[parameters],
    )

    model_selector.change(
        fn=ui_update_fn("llm", "selected_model"),
        inputs=[parameters, model_selector], outputs=[parameters],
    )

    toggle_panel.click(
        fn=panel_visibility,
        inputs=[panel_state],
        outputs=[panel_column, panel_state]
    )


print(f"{now()} gradio is starting: http://{get_local_ip()}:{config['http']['port']}")

webui.launch(
    share=config['http']['share'],
    server_name=config['http']['host'],
    server_port=config['http']['port'],
)
