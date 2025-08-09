import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

# Streamlit page config
st.set_page_config(page_title="My Chatbot", page_icon="💬")

# Title
st.title("💬 My Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Input box at bottom
user_input = st.chat_input("Type your message...")  # This automatically clears after sending

if user_input:
    # Save user's message
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    response = llm(st.session_state.messages)

    # Save AI's response
    st.session_state.messages.append(AIMessage(content=response.content))
    with st.chat_message("assistant"):
        st.markdown(response.content)
