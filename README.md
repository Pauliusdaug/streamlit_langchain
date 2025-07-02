# Kedainiai Chat Demo using LangChain and Streamlit

This is an interactive Streamlit application that leverages LangChain, OpenAI embeddings, and a custom retrieval-augmented generation (RAG) pipeline to provide accurate, source-cited answers exclusively about **Kėdainiai**, Lithuania. It combines data from multiple trusted sources and uses a GPT-4.1-based model for question answering.

---

## Features

- **Multi-source Document Loading:** Loads content from Wikipedia, the official Kėdainiai tourism website, and a local PDF document.
- **Text Chunking & Embeddings:** Splits documents into chunks and generates embeddings for efficient semantic search.
- **Retrieval-Augmented Generation:** Answers user queries by retrieving relevant information and generating responses strictly based on the defined sources.
- **Strict Topic Filtering:** Ensures answers are only provided if questions are about Kėdainiai. Otherwise, it returns a fixed rejection message.
- **Source Citation:** Clearly cites each piece of information’s source in the response.
- **Interactive UI:** Simple, user-friendly interface built with Streamlit including a form to submit questions and expandable sections to view source documents.

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/kedainiai-chat-demo.git
   cd kedainiai-chat-demo

2. Create and activate a virtual environment (optional but recommended):
    
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install required dependencies:
    
    ```bash
    pip install -r requirements.txt

4. Prepare your environment variables:
    Create a .env file with your API token:

    SECRET=your_api_token_here

5. Place the kedainiaiEN.pdf document in the project root folder.

6. Run the Streamlit app:

    ```bash
    streamlit run app.py


Usage:

    - Open the app in your browser (typically at http://localhost:8501).
    - Enter a question about Kėdainiai in the text area.
    - Submit your question to get a detailed, source-backed response.
    - Expand the source sections to view supporting documents.


How It Works:

    1. Document Loading: The app loads data from three sources:
        - Wikipedia article on Kėdainiai
        - Official tourism website visitkedainiai.lt
        - Local PDF file kedainiaiEN.pdf
    2. Text Splitting & Embeddings: Documents are chunked and embedded for semantic search.
    3. Vector Store & Retriever: A Chroma vector store indexes the chunks. The retriever fetches the most relevant chunks based on the user query.
    4. Prompt Conditioning: A custom prompt ensures the model answers only if the query pertains to Kėdainiai, otherwise returns a strict refusal message.
    5.Response Generation: The model generates an answer citing sources, which is displayed in the Streamlit UI.


Technologies Used:

    - Streamlit — for the web interface
    - LangChain — for chaining LLM calls and document handling
    - OpenAI Embeddings & GPT-4.1 — for text embeddings and response generation
    - Chroma — for vector storage and similarity search
    - Python-dotenv — for environment variable management
    - BeautifulSoup4 — for HTML parsing


Notes:

    - Make sure you have valid API access and tokens for OpenAI models.
    - The app enforces strict answer constraints to avoid hallucination and out-of-scope replies.
    - The project is designed specifically to provide accurate info on Kėdainiai only.



