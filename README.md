# Legal-Insight-RAG (Libyan Legal Context)

## Overview
 Legal-Insight-RAG is an advanced Retrieval-Augmented Generation (RAG) system specifically designed to handle the complexities of Libyan Legal Documents. Developed for the Seela AI Technical Test, this system focuses on delivering grounded, accurate answers from specialized legal PDFs while overcoming common challenges in Arabic OCR and legal text structure.

## Key Improvements & Technical Edge
 Unlike standard RAG implementations, this project addresses specific local challenges:

 -Arabic Text Rectification: Implemented a specialized preprocessing layer using arabic-reshaper and python-bidi to fix "reversed text" issues commonly found in Libyan PDF exports (e.g., in libyan_law.pdf).

 -Legal-Specific Chunking: Used a RecursiveCharacterTextSplitter with custom separators (like "مادة", "المادة") to ensure legal articles remain intact and contextually coherent.

 -Multilingual Semantic Search: Switched to paraphrase-multilingual-MiniLM-L12-v2 embeddings, which are significantly more effective at understanding Arabic legal terminology than standard English models.

 -High-Reasoning LLM: Utilizes Llama-3.3-70b-versatile via Groq API to ensure the model can perform complex legal reasoning and handle long contexts without hallucination.

## System Architecture

```mermaid
graph TD
    A[Legal PDFs] -->|pdfplumber| B(Arabic Text Correction)
    B --> C{Text Splitter}
    C -->|Chunks| D[HuggingFace Embeddings]
    D --> E[(FAISS Vector DB)]
    
    F[User Query] --> G[Semantic Search]
    E -.->|Retrieve Context| G
    G --> H[Llama-3.3-70b LLM]
    H --> I[Professional Legal Answer]


## Technical Test Requirements Covered
- Python Implementation: Pure Python 3.x.
- Free/Open-Source LLM: Integrated via Groq (Llama-3 series).
- Methodology Focused: Clear separation between data ingestion, retrieval, and generation.
- End-to-End Execution: Script covers all 5 mandatory test questions automatically.


##  Installation & Running
 1-Clone the repository:
    git clone https://github.com/tasneem-sheeha/Legal-Insight-RAG.git
    cd Legal-Insight-RAG

 2- Install Dependencies:
    pip install pdfplumber langchain langchain-community langchain-huggingface langchain-groq faiss-cpu python-dotenv arabic-reshaper python-bidi
 
 3- Environment Setup: Create a .env file and add your Groq API Key:
    GROQ_API_KEY=your_api_key_here

 4- Run the System:
    python main.py   



## Challenges & Limitations
- Structural Inconsistency: Some legal PDFs use non-standard encoding, which we mitigated using text reshaping.
- Context Window: While Llama-3.3 has a large window, legal documents can be massive; hence, efficient chunking was prioritized.
- Hallucination Risk: Managed by strict "system prompting" that forces the model to admit when information is missing.



## Future Improvements
- Citation Engine: Adding direct page-link and PDF-highlighting for verified answers.
- Hybrid Search: Combining BM25 (keyword search) with Semantic search for better article-number retrieval.
- Fine-tuned Legal Embeddings: Training on a corpus of Libyan High Court rulings.


    
Author: Tasneem Sheeha 
Project: Seela AI Technical Task
