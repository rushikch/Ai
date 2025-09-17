# 1. Install Dependencies (uncomment if not already installed)
# !pip install langchain langchain-community langchain-text-splitters sentence-transformers transformers

from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 2. Load Embedding Model (all-MiniLM-L6-v2 is small & fast)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Load LLM Model (Mistral-7B-Instruct-v0.2; choose a model you have RAM for!)
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")  # or device='cpu'

hf_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
)
llm = HuggingFacePipeline(pipeline=hf_pipe)

# 4. Sample Data (Replace with your own docs)
docs = [
    Document(page_content="LangChain lets you build powerful AI apps with composable components and integrations."),
    Document(page_content="Mistral-7B is an open-source large language model that can be used for chat or text generation."),
    Document(page_content="Retrieval Augmented Generation (RAG) combines LLMs and external data for more accurate answers."),
    Document(page_content="Embeddings are vector representations of text used for semantic search."),
]

# 5. Split Documents (if they are long)
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = splitter.split_documents(docs)

# 6. Build Vector Store
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(splits)

# 7. Define RAG Pipeline
def rag(query):
    # Retrieve relevant docs
    retrieved_docs = vector_store.similarity_search(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    # Compose prompt
    prompt = f"""Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""
    # Generate answer
    response = llm(prompt)
    return response

# 8. Test RAG
question = "What is RAG and how does it work?"
answer = rag(question)
print(f"Q: {question}\nA: {answer}")

question2 = "What is an embedding?"
answer2 = rag(question2)
print(f"\nQ: {question2}\nA: {answer2}")