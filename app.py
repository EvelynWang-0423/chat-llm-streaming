import os

import gradio as gr

from text_generation import Client, InferenceAPIClient

openchat_preprompt = (
    "\n<human>: Hi!\n<bot>: My name is Bot, model version is 0.15, part of an open-source kit for "
    "fine-tuning new bots! I was created by Together, LAION, and Ontocord.ai and the open-source "
    "community. I am not human, not evil and not alive, and thus have no thoughts and feelings, "
    "but I am programmed to be helpful, polite, honest, and friendly.\n"
)


def get_client(model: str):
    if model == "Rallio67/joi2_20B_instruct_alpha":
        return Client(os.getenv("JOI_API_URL"))
    if model == "togethercomputer/GPT-NeoXT-Chat-Base-20B":
        return Client(os.getenv("OPENCHAT_API_URL"))
    return InferenceAPIClient(model, token=os.getenv("HF_TOKEN", None))


def get_usernames(model: str):
    """
    Returns:
        (str, str, str, str): pre-prompt, username, bot name, separator
    """
    if model == "Rallio67/joi2_20B_instruct_alpha":
        return "", "User: ", "Joi: ", "\n\n"
    if model == "togethercomputer/GPT-NeoXT-Chat-Base-20B":
        return openchat_preprompt, "<human>: ", "<bot>: ", "\n"
    return "", "User: ", "Assistant: ", "\n"


def predict(
    model: str,
    inputs: str,
    top_p: float,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    watermark: bool,
    chatbot,
    history,
):
    client = get_client(model)
    preprompt, user_name, assistant_name, sep = get_usernames(model)

    history.append(inputs)

    past = []
    for data in chatbot:
        user_data, model_data = data

        if not user_data.startswith(user_name):
            user_data = user_name + user_data
        if not model_data.startswith(sep + assistant_name):
            model_data = sep + assistant_name + model_data

        past.append(user_data + model_data.rstrip() + sep)

    if not inputs.startswith(user_name):
        inputs = user_name + inputs

    total_inputs = preprompt + "".join(past) + inputs + sep + assistant_name.rstrip()

    partial_words = ""

    for i, response in enumerate(
        client.generate_stream(
            total_inputs,
            top_p=top_p if top_p < 1.0 else None,
            top_k=top_k,
            truncate=1000,
            repetition_penalty=repetition_penalty,
            watermark=watermark,
            temperature=temperature,
            max_new_tokens=500,
            stop_sequences=[user_name.rstrip(), assistant_name.rstrip()],
        )
    ):
        if response.token.special:
            continue

        partial_words = partial_words + response.token.text
        if partial_words.endswith(user_name.rstrip()):
            partial_words = partial_words.rstrip(user_name.rstrip())
        if partial_words.endswith(assistant_name.rstrip()):
            partial_words = partial_words.rstrip(assistant_name.rstrip())

        if i == 0:
            history.append(" " + partial_words)
        elif response.token.text not in user_name:
            history[-1] = partial_words

        chat = [
            (history[i].strip(), history[i + 1].strip())
            for i in range(0, len(history) - 1, 2)
        ]
        yield chat, history


def reset_textbox():
    return gr.update(value="")


def radio_on_change(
    value: str, disclaimer, top_p, top_k, temperature, repetition_penalty, watermark
):
    if value == "togethercomputer/GPT-NeoXT-Chat-Base-20B":
        top_p = top_p.update(value=0.25)
        top_k = top_k.update(value=50)
        temperature = temperature.update(value=0.6)
        repetition_penalty = repetition_penalty.update(value=1.01)
        watermark = watermark.update(False)
        disclaimer = disclaimer.update(visible=True)
    else:
        top_p = top_p.update(value=0.95)
        top_k = top_k.update(value=4)
        temperature = temperature.update(value=0.5)
        repetition_penalty = repetition_penalty.update(value=1.03)
        watermark = watermark.update(True)
        disclaimer = disclaimer.update(visible=False)
    return disclaimer, top_p, top_k, temperature, repetition_penalty, watermark


title = """<h1 align="center">🔥Large Language Model API 🚀Streaming🚀</h1>"""
description = """Language models can be conditioned to act like dialogue agents through a conversational prompt that typically takes the form:

```
User: <utterance>
Assistant: <utterance>
User: <utterance>
Assistant: <utterance>
...
```

In this app, you can explore the outputs of multiple LLMs when prompted in this way.
"""

openchat_disclaimer = """
<div align="center">Checkout the official <a href=https://huggingface.co/spaces/togethercomputer/OpenChatKit>OpenChatKit feedback app</a> for the full experience.</div>
"""

with gr.Blocks(
    css="""#col_container {margin-left: auto; margin-right: auto;}
                #chatbot {height: 520px; overflow: auto;}"""
) as demo:
    gr.HTML(title)
    with gr.Column(elem_id="col_container"):
        model = gr.Radio(
            value="togethercomputer/GPT-NeoXT-Chat-Base-20B",
            choices=[
                "togethercomputer/GPT-NeoXT-Chat-Base-20B",
                "Rallio67/joi2_20B_instruct_alpha",
                "google/flan-t5-xxl",
                "google/flan-ul2",
                "bigscience/bloom",
                "bigscience/bloomz",
                "EleutherAI/gpt-neox-20b",
            ],
            label="Model",
            interactive=True,
        )

        chatbot = gr.Chatbot(elem_id="chatbot")
        inputs = gr.Textbox(
            placeholder="Hi there!", label="Type an input and press Enter"
        )
        disclaimer = gr.Markdown(openchat_disclaimer)
        state = gr.State([])
        b1 = gr.Button()

        with gr.Accordion("Parameters", open=False):
            top_p = gr.Slider(
                minimum=-0,
                maximum=1.0,
                value=0.25,
                step=0.05,
                interactive=True,
                label="Top-p (nucleus sampling)",
            )
            temperature = gr.Slider(
                minimum=-0,
                maximum=5.0,
                value=0.6,
                step=0.1,
                interactive=True,
                label="Temperature",
            )
            top_k = gr.Slider(
                minimum=1,
                maximum=50,
                value=50,
                step=1,
                interactive=True,
                label="Top-k",
            )
            repetition_penalty = gr.Slider(
                minimum=0.1,
                maximum=3.0,
                value=1.01,
                step=0.01,
                interactive=True,
                label="Repetition Penalty",
            )
            watermark = gr.Checkbox(value=False, label="Text watermarking")

    model.change(
        lambda value: radio_on_change(
            value, disclaimer, top_p, top_k, temperature, repetition_penalty, watermark
        ),
        inputs=model,
        outputs=[disclaimer, top_p, top_k, temperature, repetition_penalty, watermark],
    )

    inputs.submit(
        predict,
        [
            model,
            inputs,
            top_p,
            temperature,
            top_k,
            repetition_penalty,
            watermark,
            chatbot,
            state,
        ],
        [chatbot, state],
    )
    b1.click(
        predict,
        [
            model,
            inputs,
            top_p,
            temperature,
            top_k,
            repetition_penalty,
            watermark,
            chatbot,
            state,
        ],
        [chatbot, state],
    )
    b1.click(reset_textbox, [], [inputs])
    inputs.submit(reset_textbox, [], [inputs])

    gr.Markdown(description)
    demo.queue(concurrency_count=16).launch(debug=True)
