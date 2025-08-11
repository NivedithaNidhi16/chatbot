import gradio as gr
from huggingface_hub import InferenceClient
import os
import time
from datetime import datetime

HF_TOKEN = os.environ.get("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def chatbot(user_input, history):
    if history is None:
        # Add initial greeting with timestamp
        history = [{"role": "assistant", "content": "Hello! I'm Niveditha's AI chatbot. How can I help you today?", "timestamp": get_timestamp()}]

    # Add user message with timestamp
    history.append({"role": "user", "content": user_input, "timestamp": get_timestamp()})

    # Prepare messages for the API (remove timestamp for API)
    messages = [{"role": "system", "content": "You are a helpful AI chatbot."}]
    # Include conversation but strip timestamp for the model input
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Call Hugging Face API
    response = client.chat_completion(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=messages,
        max_tokens=256
    )

    bot_reply = response.choices[0].message["content"]
    # Add bot reply with timestamp
    history.append({"role": "assistant", "content": bot_reply, "timestamp": get_timestamp()})

    return history, ""  # clear input

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Niveditha's AI Chatbot with Timestamp and Typing Indicator")

    chatbot_ui = gr.Chatbot(elem_id="chatbot").style(height=400)
    msg = gr.Textbox(label="Type your message", placeholder="Ask me something...")
    clear = gr.Button("Clear Chat")
    state = gr.State(None)

    # This function adds typing indicator by showing a temporary message
    def submit_message(user_input, history):
        if history is None:
            history = []
        # Add a temporary "typing" message from bot
        history.append({"role": "assistant", "content": "_Bot is typing..._", "timestamp": ""})
        # Update UI with typing message
        yield history, ""

        # Remove typing message before getting actual reply
        history.pop()

        # Call the main chatbot function to get real reply
        new_history, _ = chatbot(user_input, history)
        yield new_history, ""

    msg.submit(submit_message, [msg, state], [chatbot_ui, msg])
    clear.click(lambda: ([{"role": "assistant", "content": "Hello! I'm Niveditha's AI chatbot. How can I help you today?", "timestamp": get_timestamp()}], ""), None, [chatbot_ui, msg])

    # Custom rendering of messages with timestamp
    def format_messages(history):
        # Convert history into list of tuples for Gradio Chatbot
        formatted = []
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            timestamp = msg.get("timestamp", "")
            # Format message with timestamp
            formatted.append((f"{content}\n\n*{timestamp}*", None if role == "user" else "🤖"))
        return formatted

    # Update chat UI with formatted messages
    state.change(lambda h: format_messages(h), state, chatbot_ui)

demo.launch()
