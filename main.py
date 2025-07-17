#!/usr/bin/env python3
import os, argparse, json, copy # time, shutil
from pathlib import Path
os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = "True"

#import sys
#root = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(Path(root)/"src")

from src import embed, local_llms, chat
from src.utils import now

import yaml # uuid, litellm
import gradio as gr


#### 1. configuration
parser = argparse.ArgumentParser(
    description="parse commandline arguments",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--app", help="app name", default="LLM-RAG")
parser.add_argument("--version", help="app version", default="0.1.3")
parser.add_argument("--config", help="config path", default=Path("configs") / "local.yaml")

parser.add_argument("--prompt", help="prompt name in configs/prompts.yaml", default="default")
parser.add_argument("--rag", help="enable rag", action="store_true")
parser.add_argument("--reranker", help="enable reranker", action="store_true")

parser.add_argument("--host", help="http listening host", default="127.0.0.1")
parser.add_argument("--port", help="http listening port", type=int, default=7861)
parser.add_argument("--share", help="gradio share", action="store_true")
#parser.add_argument("--debug", help="debug mode", action="store_true")
parser.add_argument("--mode", help="running mode", default="dev")

args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

with open(Path("configs") / "prompts.yaml", 'r') as f:
    prompts = yaml.safe_load(f)

# static parameters
config['app'] = { "name": args.app, "version": args.version, "mode": args.mode }

# emoj: 📚, 🤖, 🧑, 📝, 👀, ✨, 📄, 💬, 🔍, 🐦‍⬛, 🦉, 🪶, 👤, 🎯
config['emoj'] = {
    "user": "👤",
    "ai": "🎯",
    "rag": "📝",
}

# args.app.replace(" ", "-")
config['http']['upload_dir'] = Path("data") / "uploads"
config['http']['share'] = args.share
config['http']['host'] = args.host
config['http']['port'] = args.port

config['reranker']['enabled'] = args.reranker

# dynamic parameters
config['llm'] = prompts[args.prompt]
config['llm']['system_prompt'] =  config['llm']['system_prompt'].strip()
config['llm']['user_prompt'] =  config['llm']['user_prompt'].strip()
#config['llm']['temperature'] = 0.7
#config['llm']['max_tokens'] = 1000

config['llm']['stream'] = True
_model_choices = [f"{v['provider']}/{v['model']}" for v in config['llm_models']]
config['llm']['model_choices'] = _model_choices
config['llm']['selected_model'] = _model_choices[0]

config['rag'] = { "enabled": args.rag, "verbose": False }

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
    d = config['reranker']
    reranker = {
        "model": os.path.basename(d['model']), "enabled": d['enabled'],
    }

    if mode != "dev":
        parameters = {
            "reranker": { "enabled": reranker['enabled'] },
        }
        return "**Parameters**: " + json.dumps(parameters)

    d = config['embedding']
    embedding = { "provider": d['provider'], "model": d['model'] }

    d = config['qdrant']
    vector_db = { "collection": d['collection'] }

    parameters = {
        "reranker": reranker, "embedding": embedding, "vector_db": vector_db,
    }

    #return "Parameters\n" + "\n".join(strs)
    return "**Parameters**: " + json.dumps(parameters)


def panel_visibility(current_state):
    new_state = not current_state
    return gr.update(visible=new_state), new_state


#### 4. run
with gr.Blocks(
    title=f"{config['app']['name']}:{config['app']['mode']}-{config['app']['version']}",
    css=config['http'].get('css'), js=config['http'].get('js'),
) as webui:
    #param_info = {
    #    "temperature": {"type": "float", "description": "Creativity level", "default": 0.7},
    #    "max_tokens": {"type": "int", "description": "Max tokens in response", "default": 256}
    #}

    upload_file_types = config['http']['upload_file_types']

    parameters = gr.State({
        "app": copy.deepcopy(config['app']),
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
                lines=12, max_lines=12,
                value=config['llm']['user_prompt'],
            )

            with gr.Row():
                #clear_button = gr.Button("Clear", scale=1)
                with gr.Row():
                    model_selector = gr.Dropdown(
                        interactive=True, show_label=True, label="Select a model",
                        value=config['llm']['selected_model'],
                        choices=config['llm']['model_choices'],
                        elem_id="model-selector", scale=2,
                    )

                    stream_checkbox = gr.Checkbox(label="stream", value=config['llm']['stream'])

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
                    rag_verbose = gr.Checkbox(label="verbose", value=config['rag']['verbose'])

                    top_n_slider = gr.Slider(
                        label="top_n",
                        value=config['qdrant']['top_n'], minimum=1, maximum=30, step=1,
                    )

                    score_threshold_slider = gr.Slider(
                        label="score_threshold",
                        value=config['qdrant']['score_threshold'],
                        minimum=0.1, maximum=1.0, step=0.01,
                    )


            uploaded_files = gr.File(
                label=f"Upload docs: {', '.join(upload_file_types)}",
                file_types=upload_file_types, file_count="multiple",
            )


        with gr.Column(scale=7, elem_id="chat-column"):
            with gr.Row(elem_id="chat-header"):
                toggle_panel = gr.Button("Panel", elem_id="toggle-panel")

                with gr.Column(scale=8):
                    gr.Markdown(display_parameters(config['app']['mode']))
                #gr.ParamViewer(value=param_info)

            # height=600
            chatbot = gr.Chatbot(label="AI Assistant", type='messages', elem_id="my-chatbot")

            gr.ChatInterface(
                chat.chat_fn, type="messages", chatbot=chatbot,
                additional_inputs=[uploaded_files, parameters],
                additional_outputs=[chatbot],
                multimodal=False, autofocus=True,
            )

    #system_prompt_input.change(
    #    fn=lambda value: update_llm_textbox("system_prompt", value),
    #    inputs=system_prompt_input, outputs=system_prompt_input,
    #)

    ####
    system_prompt_input.change(
        fn=chat.ui_update_str("llm", "system_prompt"),
        inputs=[parameters, system_prompt_input], outputs=[],
    )

    user_prompt_input.change(
        fn=chat.ui_update_str("llm", "user_prompt"),
        inputs=[parameters, user_prompt_input], outputs=[],
    )

    ####
    stream_checkbox.change(
        fn=chat.ui_update_fn("llm", "stream"),
        inputs=[parameters, stream_checkbox], outputs=[],
    )

    max_tokens_input.change(
        fn=chat.ui_update_fn("llm", "max_tokens", 20),
        inputs=[parameters, max_tokens_input], outputs=[],
    )

    temperature_slider.change(
        fn=chat.ui_update_fn("llm", "temperature"),
        inputs=[parameters, temperature_slider], outputs=[],
    )

    ####
    rag_checkbox.change(
        fn=chat.ui_update_fn("rag", "enabled"),
        inputs=[parameters, rag_checkbox], outputs=[],
    )

    rag_verbose.change(
        fn=chat.ui_update_fn("rag", "verbose"),
        inputs=[parameters, rag_verbose], outputs=[],
    )

    top_n_slider.change(
        fn=chat.ui_update_fn("qdrant", "top_n"),
        inputs=[parameters, top_n_slider], outputs=[],
    )

    score_threshold_slider.change(
        fn=chat.ui_update_fn("qdrant", "score_threshold"),
        inputs=[parameters, score_threshold_slider], outputs=[],
    )

    model_selector.change(
        fn=chat.ui_update_fn("llm", "selected_model"),
        inputs=[parameters, model_selector], outputs=[],
    )

    toggle_panel.click(
        fn=panel_visibility,
        inputs=[panel_state],
        outputs=[panel_column, panel_state]
    )


webui.launch(
    share=config['http']['share'],
    server_name=config['http']['host'],
    server_port=config['http']['port'],
)
