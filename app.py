import gradio as gr
import json
from huggingface_hub import InferenceClient

# Hugging Face API Client

import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

# Store loaded JSON
json_data = None

def upload_json(file):
    """Load JSON file from Gradio upload."""
    global json_data
    if file is None:
        return "⚠ No file uploaded yet."
    try:
        with open(file.name, "r") as f:
            json_data = json.load(f)
        return f"✅ Loaded JSON file: {file.name}"
    except Exception as e:
        return f"❌ Error loading file: {e}"

def chatbot(user_input, history):
    """Chat with Mistral using optional JSON context."""
    global json_data

    if history is None:
        history = []

    # Base system message
    messages = [{"role": "system", "content": "You are a helpful AI chatbot."}]

    # Add JSON context if loaded
    if json_data:
        messages.append({
            "role": "system",
            "content": f"Here is extra JSON context:\n{json.dumps(json_data, indent=2)}"
        })

    # Add history + new user message
    for human, bot in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": bot})
    messages.append({"role": "user", "content": user_input})

    # Call Mistral
    response = client.chat_completion(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=messages,
        max_tokens=500
    )

    bot_reply = response.choices[0].message["content"]
    history.append((user_input, bot_reply))
    return history, ""  # clear input

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📄 Mistral JSON-Aware Chatbot")
    
    file_upload = gr.File(label="Upload JSON File", file_types=[".json"])
    upload_status = gr.Textbox(label="Upload Status", interactive=False)
    file_upload.change(upload_json, inputs=file_upload, outputs=upload_status)
    
    chatbot_ui = gr.Chatbot()
    msg = gr.Textbox(label="Your message")
    send_btn = gr.Button("Send")
    
    send_btn.click(chatbot, inputs=[msg, chatbot_ui], outputs=[chatbot_ui, msg])

demo.launch()
