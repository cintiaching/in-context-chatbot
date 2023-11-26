import tempfile
import pathlib
import streamlit as st
from document_chatbot.llama2_model import init_llama2_13b_llm
from document_chatbot.openai_model import init_openai_model, TokenCounter
from document_chatbot.rag import init_qa_chain
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


def generate_response(input_text):
    response = qa_chain.run(input_text)
    st.info(response)
    return response


st.title("Document Chatbot")
st.text("The Document Chatbot using Llama2 13B or gpt-3.5-turbo.\n👈 Please select a model and upload a docx/pdf files.")

# keep track of token used
token_counter = TokenCounter()

# select model
with st.sidebar:
    option = st.selectbox(
        "Model",
        ("Llama2 13B", "gpt-3.5-turbo")
    )

    if option == "gpt-3.5-turbo":
        llm = init_openai_model()

    elif option == "Llama2 13B":
        llm = init_llama2_13b_llm()

    # upload file
    uploaded_file = st.file_uploader("Upload a docx/pdf files as the context of the chatbot:")
    if uploaded_file is not None:
        tmp_location = tempfile.TemporaryDirectory()
        tmp_file_path = pathlib.Path(tmp_location.name) / uploaded_file.name
        with open(tmp_file_path, 'wb') as output_temporary_file:
            output_temporary_file.write(uploaded_file.read())

        DOCUMENT_PATH = tmp_location
        if uploaded_file.name.endswith("pdf"):
            doc_type = "pdf"
        elif uploaded_file.name.endswith("docx"):
            doc_type = "docx"
        else:
            raise ValueError(f"The format of uploaded file is not supported")

        qa_chain = init_qa_chain(
            str(tmp_file_path),
            llm,
            chunk_size=500,
            chunk_overlap=20,
            prompt=None,
            doc_type=doc_type,
        )

with st.form("my_form"):
    text = st.text_area("Ask a question:", "")
    input_token = llm.get_num_tokens(text)
    token_counter.add(input_token)
    submitted = st.form_submit_button("Submit")
    if submitted:
        response = generate_response(text)
        output_token = llm.get_num_tokens(response)
        token_counter.add(output_token)

        with st.sidebar:
            st.divider()
            st.text(
                f"Token used for input: {input_token}\n"
                f"Token used for output: {output_token}\n"
                f"Total Token used: {token_counter.total_token_used()}\n"
            )
