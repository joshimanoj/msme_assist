# msme_assist

Chatbot to assist MSMEs (Micro, Small & Medium Enterprises) with government schemes and business growth guidance.

## Features

- **Scheme Discovery**: Semantic search over government schemes using FAISS vector indexing
- **Business Growth Guidance**: AI-powered advice tailored to MSME needs
- **Conversational Interface**: Natural language interaction via Streamlit
- **Curated Data**: Pre-processed scheme database with embeddings for fast retrieval

## Tech Stack

- Python, Streamlit
- FAISS (vector similarity search)
- OpenAI / LLM APIs

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and configure API keys
3. Run: `streamlit run app.py`
