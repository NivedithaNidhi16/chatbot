import gradio as gr
from huggingface_hub import InferenceClient
import os

# Store API key securely
HF_TOKEN = os.environ.get("HF_TOKEN")

# Create inference client
client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=HF_TOKEN)

# Chat function
def chatbot(user_input, history):
    # Format conversation
    messages = [{"role": "system", "content": "You are a helpful AI chatbot."}]
    for h in history:
        messages.append({"role": "user", "content": h[0]])
        messages.append({"role": "assistant", "content": h[1]])
    messages.append({"role": "user", "content": user_input})

    # Get model response
    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=messages,
        max_tokens=256
    )

    bot_reply = response.choices[0].message["content"]
    history.append((user_input, bot_reply))
    return history, ""  # "" clears the input box

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Niveditha's AI Chatbot")
    chatbot_ui = gr.Chatbot()
    msg = gr.Textbox(label="Type your message", placeholder="Say something...", lines=1)
    clear = gr.Button("Clear Chat")

    state = gr.State([])

    msg.submit(chatbot, [msg, state], [chatbot_ui, msg]).then(
        lambda hist: hist, None, state
    )
    clear.click(lambda: ([], ""), None, [chatbot_ui, msg])

# Launch app
demo.launch()
