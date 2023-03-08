import os

import gradio as gr

from text_generation import Client, InferenceAPIClient


def get_client(model: str):
    if model == "Rallio67/joi2_20B_instruct_alpha":
        return Client(os.getenv("API_URL"))
    return InferenceAPIClient(model, token=os.getenv("HF_TOKEN", None))


def get_usernames(model: str):
    if model == "Rallio67/joi2_20B_instruct_alpha":
        return "User: ", "Joi: "
    return "User: ", "Assistant: "


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
    user_name, assistant_name = get_usernames(model)

    history.append(inputs)

    past = []
    for data in chatbot:
        user_data, model_data = data

        if not user_data.startswith(user_name):
            user_data = user_name + user_data
        if not model_data.startswith("\n\n" + assistant_name):
            model_data = "\n\n" + assistant_name + model_data

        past.append(user_data + model_data + "\n\n")

    if not inputs.startswith(user_name):
        inputs = user_name + inputs

    total_inputs = "".join(past) + inputs + "\n\n" + assistant_name
    # truncate total_inputs
    total_inputs = total_inputs[-1000:]

    partial_words = ""

    for i, response in enumerate(client.generate_stream(
            total_inputs,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            watermark=watermark,
            temperature=temperature,
            max_new_tokens=500,
            stop_sequences=[user_name.rstrip(), assistant_name.rstrip()],
    )):
        if response.token.special:
            continue

        partial_words = partial_words + response.token.text
        if partial_words.endswith(user_name.rstrip()):
            partial_words = partial_words.rstrip(user_name.rstrip())
        if partial_words.endswith(assistant_name.rstrip()):
            partial_words = partial_words.rstrip(assistant_name.rstrip())

        if i == 0:
            history.append(" " + partial_words)
        else:
            history[-1] = partial_words

        chat = [
            (history[i], history[i + 1]) for i in range(0, len(history) - 1, 2)
        ]
        yield chat, history


def reset_textbox():
    return gr.update(value="")


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

with gr.Blocks(
        css="""#col_container {width: 1000px; margin-left: auto; margin-right: auto;}
                #chatbot {height: 520px; overflow: auto;}"""
) as demo:
    gr.HTML(title)
    with gr.Column(elem_id="col_container"):
        model = gr.Radio(
            value="Rallio67/joi2_20B_instruct_alpha",
            choices=[
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
        state = gr.State([])
        b1 = gr.Button()

        with gr.Accordion("Parameters", open=False):
            top_p = gr.Slider(
                minimum=-0,
                maximum=1.0,
                value=0.95,
                step=0.05,
                interactive=True,
                label="Top-p (nucleus sampling)",
            )
            temperature = gr.Slider(
                minimum=-0,
                maximum=5.0,
                value=0.5,
                step=0.1,
                interactive=True,
                label="Temperature",
            )
            top_k = gr.Slider(
                minimum=1,
                maximum=50,
                value=4,
                step=1,
                interactive=True,
                label="Top-k",
            )
            repetition_penalty = gr.Slider(
                minimum=0.1,
                maximum=3.0,
                value=1.03,
                step=0.01,
                interactive=True,
                label="Repetition Penalty",
            )
            watermark = gr.Checkbox(value=True, label="Text watermarking")

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
    demo.queue().launch(debug=True)
