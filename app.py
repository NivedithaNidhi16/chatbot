import gradio as gr
from huggingface_hub import InferenceClient
import os
from datetime import datetime

HF_TOKEN = os.environ.get("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Main chatbot function
def chatbot(user_input, history):
    # Add user input with timestamp
    history.append({"role": "user", "content": user_input, "timestamp": get_timestamp()})

    # Prepare messages for API call (strip timestamps)
    messages = [{"role": "system", "content": "You are a helpful AI chatbot."}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    response = client.chat_completion(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=messages,
        max_tokens=256
    )

    bot_reply = response.choices[0].message["content"]
    # Add bot reply with timestamp
    history.append({"role": "assistant", "content": bot_reply, "timestamp": get_timestamp()})

    return history, ""

# Typing indicator wrapper
def submit_message(user_input, history):
    # Add temporary typing message
    history.append({"role": "assistant", "content": "_Bot is typing..._", "timestamp": ""})
    yield history, ""

    # Remove typing message
    history.pop()

    new_history, _ = chatbot(user_input, history)
    yield new_history, ""

# Format messages for display with timestamps
def format_messages(history):
    formatted = []
    if history is None:
        return formatted
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        timestamp = msg.get("timestamp", "")
        if role == "user":
            formatted.append((f"{content}\n\n*{timestamp}*", None))
        else:
            formatted.append((f"🤖 {content}\n\n*{timestamp}*", None))
    return formatted

# Initial greeting to start the chat with
initial_greeting = [{"role": "assistant", "content": "Hello! I'm Niveditha's AI chatbot. How can I help you today?", "timestamp": get_timestamp()}]

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Niveditha's AI Chatbot")

    chatbot_ui = gr.Chatbot(elem_id="chatbot", type="messages")
    msg = gr.Textbox(label="Type your message", placeholder="Ask me something...")
    clear = gr.Button("Clear Chat")
    state = gr.State(initial_greeting)  # initialize with greeting

    # On submit, call submit_message, keep history in state
    msg.submit(submit_message, [msg, state], [chatbot_ui, msg])
    # Clear chat resets to initial greeting
    clear.click(lambda: (initial_greeting, ""), None, [chatbot_ui, msg])
    # Format chat UI whenever history changes
    state.change(format_messages, state, chatbot_ui)

demo.launch()
