import streamlit as st
from document_chatbot.llama2 import init_llama2_13b_llm, init_qa_chain
from document_chatbot.utils import DOCUMENT_PATH


def generate_response(input_text):
    st.info(qa_chain.run(input_text))


# TODO: file upload option https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader
llm = init_llama2_13b_llm()
qa_chain = init_qa_chain(
    DOCUMENT_PATH,
    llm,
    chunk_size=500,
    chunk_overlap=20,
    prompt=None,
    doc_type="pdf",
)

st.title("Document Chatbot")
st.text("The New Staff Handbook Chatbot using Llama2 13B.")


with st.form("my_form"):
    text = st.text_area("Enter text:", "Tell me about Sick Leave")
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)
