import streamlit as st
from rag_llama2_13b import init_qa_chain, ask

st.title("🦜🔗 Quickstart App")

path = "./data/Employment+Handbook_0000020125.pdf"
qa_chain_full_book = init_qa_chain(
    path,
    chunk_size=500,
    chunk_overlap=20,
    prompt=None,
    doc_type="pdf",
)


def generate_response(input_text):
    st.info(ask(input_text, qa_chain_full_book))


with st.form('my_form'):
    text = st.text_area('Enter text:', 'Tell me about Sick Leave')
    submitted = st.form_submit_button('Submit')
    if submitted:
        generate_response(text)
