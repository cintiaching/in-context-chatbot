import os
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import AzureOpenAIEmbeddings  # OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough

"""
openai.PermissionDeniedError: Error code: 403 - {'error': {'code': 'AccessDenied', 'message': 'Public access is disabled. Please configure private endpoint.'}}
https://learn.microsoft.com/en-us/azure/ai-services/cognitive-services-virtual-networks?tabs=portal
"""

# IDEA: can add a prompt template telling it to say "irrelevant" if it is not in the docs
# IDEA: refer to article by sophia

# set api key
os.environ["OPENAI_API_TYPE"] = "azure"
# os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://llm-openai-rnd.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = ""

# init model
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
)
# load document
loader = UnstructuredWordDocumentLoader(
    "New Staff Handbook Q&A.docx", mode="elements", strategy="fast",
)
docs = loader.load()

# Split the Document into chunks for embedding and vector storage.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(docs)

# store the splits to look up later
vectorstore = Chroma.from_documents(documents=all_splits, embedding=AzureOpenAIEmbeddings())


def rag_chatbot(question):
    # retrieve relevant splits for any question using similarity search
    retriever = vectorstore.similarity_search(question)
    # Distill the retrieved documents into an answer using an LLM/Chat model
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | question | llm
    # generate
    rag_chain.invoke(question)


rag_chatbot("why is there a new staff handbook?")
