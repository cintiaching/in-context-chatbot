# In Context Chatbot
Retrieval Augmented Generation (RAG) is used. Proposed algorithm are shown in the flowchart.
```mermaid
%%{
    init: {
        'theme': 'base',
        'themeVariables': {
            'primaryColor': '#fff',
            'primaryTextColor': '#000000',
            'primaryBorderColor': '#000000',
            'lineColor': '#799c6e',
            'secondaryColor': '#006100',
            'tertiaryColor': '#ecfae8'
        }
    }
}%%
flowchart LR
    subgraph Ingestion Stage
    J([Input Document#40;s#41;]) --> K[Data Cleansing]
    K --> M[Chunking]
    M --> N[Embedding Model]
    M --> O[Metadata of the Chunks]
    O --> Vectorstore[(Vectorstore)]
    N --> Vectorstore
    end

    Vectorstore --> Retriever
    Query([Query]) --> B[Query Transformation]

    subgraph Retrieval Stage
    B --> Retriever[Retriever]
    end

    Retriever --> D[Post-processing + Reranking]
    D --> E(Retrieved Context)
    E --> LLM[LLM#40;s#41;]

    Query --> G[(History Queries)]
    subgraph Chatbot Memory
    G --> H[Compression / Summarization]
    end

    H --> Retriever
    I([Predefined Prompt]) --> LLM
    Query --- I
    LLM --> F([Generate Answer])
```
## Usage
Run `streamlit run document_chatbot_app.py`, a local Streamlit server will spin up and the app will open in a new tab in the default web browser.
