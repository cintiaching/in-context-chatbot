import streamlit as st
import time

from chatbots.staff_q_and_a.staff_q_and_a import StaffQAChatbot

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def generate_response(input_text):
    response = qa_chain.run(input_text)
    st.info(response)
    return response


st.title("Staff Q&A Chatbot")
st.text("The Staff Q&A Chatbot is using Llama2 13B or gpt-3.5-turbo.\n👈 Please select a model.")

# select model
option = st.selectbox(
    "Model",
    ("Llama2 13B", "gpt-3.5-turbo")
)

if option == "gpt-3.5-turbo":
    model_name = "openai"
elif option == "Llama2 13B":
    model_name = "llama2_13b"

if option is not None:
    staff_qa_chatbot = StaffQAChatbot(
        doc_path="data/New Staff Handbook Q&A.docx",
        model_name=model_name,
    )
    qa_chain = staff_qa_chatbot.qa_chain()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = generate_response(prompt)
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
