import gradio as gr
from huggingface_hub import InferenceClient
import os

# Load API key
HF_TOKEN = os.environ.get("HF_TOKEN")

# Create client
client = InferenceClient(token=HF_TOKEN)

# Chat function
def chatbot(user_input, history):
    if history is None:
        history = []

    # Convert Gradio's messages format into HF's messages format
    messages = [{"role": "system", "content": "You are a helpful AI chatbot."}]
    messages.extend(history)  # history is already in role/content dicts
    messages.append({"role": "user", "content": user_input})

    # Call Hugging Face API for chat completion
    response = client.chat_completion(
        model="tiiuae/falcon-7b-instruct",
        messages=messages,
        max_tokens=500
    )

    bot_reply = response.choices[0].message["content"]

    # Append conversation to history
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": bot_reply})

    return history, ""  # Clear input

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Niveditha's AI Chatbot")

    chatbot_ui = gr.Chatbot(type="messages")
    msg = gr.Textbox(label="Type your message", placeholder="Ask me something...")
    clear = gr.Button("Clear Chat")

    state = gr.State([])

    msg.submit(chatbot, [msg, state], [chatbot_ui, msg])
    clear.click(lambda: ([], ""), None, [chatbot_ui, msg])

# Launch app
demo.launch()
