import os
import json
from tkinter import Tk, filedialog
from huggingface_hub import InferenceClient

# Load Hugging Face API token from environment
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

# Global variable to store JSON data
json_data = None

def pick_json_file():
    """Opens file picker and loads JSON into global variable."""
    global json_data
    Tk().withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
    if file_path:
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        print(f"Loaded JSON: {file_path}")
    else:
        print("No JSON selected.")

def chatbot(user_input, history):
    global json_data
    
    if history is None:
        history = []
    
    # If no JSON loaded yet, ask to pick one (optional)
    if json_data is None:
        pick_json_file()

    # Prepare base messages
    messages = [{"role": "system", "content": "You are a helpful AI chatbot."}]
    messages.extend(history)
    
    # Add JSON data as context if loaded
    if json_data:
        messages.append({
            "role": "system",
            "content": f"Here is additional context in JSON format:\n{json.dumps(json_data, indent=2)}"
        })

    # Add user question
    messages.append({"role": "user", "content": user_input})

    # Call Hugging Face Chat Completion API
    response = client.chat_completion(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=messages,
        max_tokens=500
    )

    bot_reply = response.choices[0].message["content"]

    # Update history
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": bot_reply})

    return history, ""  # Clear input box

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
