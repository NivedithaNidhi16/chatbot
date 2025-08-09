import gradio as gr
from huggingface_hub import InferenceClient
import os

# Load API key from environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")

# Create Hugging Face inference client
client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=HF_TOKEN)

# Chat function
def chatbot(user_input, history):
    if history is None:
        history = []

    # Start with system prompt
    messages = [{"role": "system", "content": "You are a helpful AI chatbot."}]

    # Add previous messages from history
    for msg in history:
        messages.append(msg)

    # Add new user message
    messages.append({"role": "user", "content": user_input})

    # Get model response
    response = client.chat(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=messages,
        max_tokens=256
    )

    bot_reply = response.choices[0].message["content"]

    # Append AI response to history
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": bot_reply})

    return history, ""  # Clear input box

# Build Gradio UI
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
