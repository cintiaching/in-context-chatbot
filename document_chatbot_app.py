import tempfile
import pathlib
import streamlit as st
from document_chatbot.llama2 import init_llama2_13b_llm, init_qa_chain


def generate_response(input_text):
    st.info(qa_chain.run(input_text))


llm = init_llama2_13b_llm()

st.title("Document Chatbot")
st.text("The Document Chatbot using Llama2 13B. Please upload a docx/pdf files")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    tmp_location = tempfile.TemporaryDirectory()
    tmp_file_path = pathlib.Path(tmp_location.name) / uploaded_file.name
    with open(tmp_file_path, 'wb') as output_temporary_file:
        output_temporary_file.write(uploaded_file.read())

    DOCUMENT_PATH = tmp_location
    if uploaded_file.name.endswith("pdf"):
        doc_type = "pdf"
    elif uploaded_file.name.endswith("docx"):
        doc_type = "pdf"
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
        text = st.text_area("Enter text:", "")
        submitted = st.form_submit_button("Submit")
        if submitted:
            generate_response(text)
