#!/usr/bin/env python3
import re

import rag
from src.utils import now


def update_value(sub, key):
    def fn(parameters, value):
        print(f"<-- update {sub} {key}: {value}")
        parameters[sub][key] = value
        return parameters

    return fn

def update_value_min(sub, key, min_val=None):
    def fn(parameters, value):
        if min_val and value < min_val:
            value = min_val

        print(f"<-- update_value_min {sub} {key}: {value}")
        parameters[sub][key] = value
        return parameters

    return fn


def handle_user_input(user_input, uploaded_files, parameters):
    if not parameters['rag']['enabled'] or not uploaded_files:
        return ([], [], user_input)

    docs_files, rag_outputs = rag.rag_query_docs(
        user_input, uploaded_files, parameters,
    )

    if len(rag_outputs) == 0:
        return (docs_files, [], user_input)

    # print(f"<-- rag outputs: {rag_outputs}")
    texts = [f"#### {i+1}. {v}" for i, v in enumerate(rag_outputs)]

    user_prompt = parameters['llm']['user_prompt']
    user_input = f"{user_prompt}".format(input=user_input, context="\n\n".join(texts))

    return (docs_files, rag_outputs, user_input)


def chat_func(history, user_input, uploaded_files, system_prompt, user_prompt, parameters):
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

    docs_files, rag_outputs, user_input = handle_user_input(user_input, uploaded_files, parameters)

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
