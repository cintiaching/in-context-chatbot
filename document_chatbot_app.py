import streamlit as st
from document_chatbot.rag_llama2_13b import init_llama2_13b_llm, init_qa_chain

st.title("Staff Handbook Chatbot")
st.text("The New Staff Handbook Chatbot using Llama2 13B.")

# TODO: file upload option https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader
path = "./data/Employment+Handbook_0000020125.pdf"

llm = init_llama2_13b_llm()
qa_chain = init_qa_chain(
    path,
    llm,
    chunk_size=500,
    chunk_overlap=20,
    prompt=None,
    doc_type="pdf",
)


def generate_response(input_text):
    st.info(qa_chain.run(input_text))


with st.form("my_form"):
    text = st.text_area("Enter text:", "Tell me about Sick Leave")
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)
