import os

import streamlit as st
import tempfile
import pathlib
import time

from chatbots.llm.llm import LLMs
from chatbots.rag.document_a import DocumentAChatbot
from chatbots.rag.document_b import DocumentBChatbot
from chatbots.rag.general_document import GeneralChatbot

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def generate_response(input_text):
    response = qa_chain.invoke(input_text)
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
    ("Document A", "Document B", "Upload Your Document")
)

if selected_doc == "Document B":
    document_b = DocumentBChatbot(
        doc_path=os.environ.get("DOCUMENT_B_PATH"),
        model_name=LLMs(selected_model),
        chatbot_name="document_b",
    )
    qa_chain = document_b.qa_chain()
elif selected_doc == "Document A":
    document_a = DocumentAChatbot(
        doc_path=os.environ.get("DOCUMENT_A_PATH"),
        model_name=LLMs(selected_model),
        chatbot_name="document_a",
    )
    qa_chain = document_a.qa_chain()
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
            collection_name=uploaded_file.name,
            persist_directory=f"./data/{uploaded_file.name}_vectorstore"
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
            result = assistant_response["result"]
            source_documents = assistant_response["source_documents"]
            # Simulate stream of response with milliseconds delay
            print(assistant_response)
            for chunk in result.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        # show retrieved contexts
        with st.sidebar:
            st.header("Retrieved Contexts (Ranked by Relevancy)")
            for i, sd in enumerate(source_documents):
                st.markdown(str(i + 1) + ".")
                st.markdown(sd.page_content)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
