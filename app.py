import gradio as gr
from huggingface_hub import InferenceClient
import os
from datetime import datetime

# Load Hugging Face token from environment variable
HF_TOKEN = os.environ.get("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def chatbot(user_input, history):
    if history is None:
        # Initial greeting from bot with timestamp
        history = [{"role": "assistant", "content": "Hello! I'm Niveditha's AI chatbot. How can I help you today?", "timestamp": get_timestamp()}]

    # Add user message with timestamp
    history.append({"role": "user", "content": user_input, "timestamp": get_timestamp()})

    # Prepare messages for API (strip timestamps)
    messages = [{"role": "system", "content": "You are a helpful AI chatbot."}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Call Hugging Face chat completion API
    response = client.chat_completion(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=messages,
        max_tokens=256
    )

    bot_reply = response.choices[0].message["content"]
    # Add bot reply with timestamp
    history.append({"role": "assistant", "content": bot_reply, "timestamp": get_timestamp()})

    return history, ""  # clear user input box

# Function to add typing indicator before calling chatbot
def submit_message(user_input, history):
    if history is None:
        history = []
    # Add temporary typing message
    history.append({"role": "assistant", "content": "_Bot is typing..._", "timestamp": ""})
    yield history, ""

    # Remove typing message before real reply
    history.pop()

    new_history, _ = chatbot(user_input, history)
    yield new_history, ""

# Format messages for Gradio Chatbot component with timestamps
def format_messages(history):
    formatted = []
    if history is None:
        return formatted
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        timestamp = msg.get("timestamp", "")
        # Display message with timestamp below it
        if role == "user":
            formatted.append((f"{content}\n\n*{timestamp}*", None))
        else:
            formatted.append((f"🤖 {content}\n\n*{timestamp}*", None))
    return formatted

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Niveditha's AI Chatbot with Timestamp & Typing Indicator")

    chatbot_ui = gr.Chatbot(elem_id="chatbot", type="messages")
    msg = gr.Textbox(label="Type your message", placeholder="Ask me something...")
    clear = gr.Button("Clear Chat")
    state = gr.State(None)

    # Wire up submit with typing indicator
    msg.submit(submit_message, [msg, state], [chatbot_ui, msg])
    # Clear chat and reset with initial greeting
    clear.click(lambda: ([{"role": "assistant", "content": "Hello! I'm Niveditha's AI chatbot. How can I help you today?", "timestamp": get_timestamp()}], ""), None, [chatbot_ui, msg])
    # Update chatbot UI to formatted messages whenever state changes
    state.change(format_messages, state, chatbot_ui)

demo.launch()
