
# Retrieval Augmented Generation (RAG) Pipeline

This project is a simple, open-source Retrieval Augmented Generation (RAG) pipeline using HuggingFace models and LangChain. 

RAG combines the power of large language models (LLMs) with external knowledge sources for more accurate and context-aware answers.

## Features
- **Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` for fast, high-quality text embeddings.
- **LLM:** Utilizes `mistralai/Mistral-7B-Instruct-v0.2` for text generation.
- **Vector Store:** In-memory vector store for semantic search.
- **Document Splitting:** Handles long documents with recursive character splitting.
- **End-to-End RAG Pipeline:** Retrieve relevant context and generate answers to user queries.

## Installation
Uncomment and run the following in your Python environment if dependencies are not installed:

```python
!pip install langchain langchain-community langchain-text-splitters sentence-transformers transformers
```

## Usage
1. Clone this repository and open `RAG_Pipeline_OpenSource.py`.
2. Ensure you have enough RAM for the selected LLM (Mistral-7B is recommended for capable machines).
3. Run the script to see a sample RAG workflow:
	- Loads embedding and LLM models
	- Splits sample documents
	- Builds a vector store
	- Answers a sample question using RAG

## Example
```
Q: What is RAG and how does it work?
A: [Generated answer from the LLM]
```

## Customization
- Replace the sample `docs` list with your own documents.
- Adjust chunk size and overlap in the text splitter as needed.
- Swap in different embedding or LLM models as required.

## References
- [LangChain Documentation](https://python.langchain.com/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [Mistral-7B Model Card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

---
*Open source RAG pipeline for rapid prototyping and experimentation.*
