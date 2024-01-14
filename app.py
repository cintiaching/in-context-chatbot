import streamlit as st
import tempfile
import pathlib
import time

from chatbots.model_choices import LLMs
from chatbots.staff_q_and_a.staff_q_and_a import StaffQAChatbot
from chatbots.staff_handbook.staff_handbook import StaffHandbookChatbot
from chatbots.general_document import GeneralChatbot

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def generate_response(input_text):
    response = qa_chain.run(input_text)
    return response


st.title("Contextual Chatbot 💡")
st.text("👋 Welcome to the Q&A chatbot! \n"
        "📚 Just input your document, and our chatbot will provide insightful answers, \n"
        "explanations, and assistance \n"
        "😊 Please select a model and select/upload a document"
)

# select model
selected_model = st.selectbox(
    "Model",
    (LLMs.GPT_3_PT_5_TURBO.value, LLMs.LLAMA2_13B.value, LLMs.MISTRAL_7B.value)
)

# select document
selected_doc = st.selectbox(
    "Document",
    ("Staff Handbook", "20 Questions in Staff Q&A", "Upload Your Document")
)

if selected_doc == "Staff Handbook":
    staff_handbook_chatbot = StaffHandbookChatbot(
        doc_path="data/Hong Kong Staff Handbook_2023 11 01 (Part A B EN)_2023 12 01_Clean.docx",
        model_name=LLMs(selected_model),
    )
    qa_chain = staff_handbook_chatbot.qa_chain()
elif selected_doc == "20 Questions in Staff Q&A":
    staff_qa_chatbot = StaffQAChatbot(
        doc_path="data/New Staff Handbook Q&A.docx",
        model_name=LLMs(selected_model),
    )
    qa_chain = staff_qa_chatbot.qa_chain()
elif selected_doc == "Upload Your Document":
    # upload option appears
    uploaded_file = st.file_uploader("Upload a docx/pdf file")
    if uploaded_file is not None:
        # loader from langchain requires path as input
        tmp_location = tempfile.TemporaryDirectory()
        tmp_file_path = pathlib.Path(tmp_location.name) / uploaded_file.name
        with open(tmp_file_path, 'wb') as output_temporary_file:
            output_temporary_file.write(uploaded_file.read())

        custom_doc_chatbot = GeneralChatbot(
            doc_path=str(tmp_file_path),
            model_name=LLMs(selected_model),
        )
        qa_chain = custom_doc_chatbot.qa_chain()

# chatbot interface
if selected_model is not None and selected_doc is not None:
    st.text("----------------------------------------------------------------------------------")
    if st.button("Clear History 🧹", type="primary"):
        st.session_state.messages = []
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is this document about?"):
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
