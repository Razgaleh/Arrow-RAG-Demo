# Standard library imports
import os 
from pathlib import Path
 
 
# LLM and embedding model imports
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import HuggingFaceEmbeddings

# Document loading and processing imports
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter

# Vector store and QA chain imports
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.messages import AIMessage, HumanMessage

# Get the current working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Single source of truth for "Public Websites (URL)" RAG mode — titles shown in the UI;
# URLs are loaded into the vector index by local_get_answer_url / nim_get_answer_url.
RAG_WEBSITE_SOURCES = (
    {
        "label": "Wikipedia — Arrow Electronics",
        "url": "https://en.wikipedia.org/wiki/Arrow_Electronics",
    },
    {
        "label": "LinkedIn — company posts",
        "url": "https://www.linkedin.com/company/arrow-electronics/posts/?feedView=all",
    },
    {
        "label": "Built In Colorado — employer profile",
        "url": "https://www.builtincolorado.com/company/arrow-electronics-inc",
    },
    {
        "label": "Bloomberg — ARW:US company profile",
        "url": "https://www.bloomberg.com/profile/company/ARW:US",
    },
)


def _rag_website_urls():
    return [entry["url"] for entry in RAG_WEBSITE_SOURCES]


def docs_folder_path() -> Path:
    """Directory containing bundled PDFs for the Local PDFs RAG mode."""
    return Path(working_dir) / "docs"


def list_docs_folder_pdfs():
    """Metadata for each ``*.pdf`` under ``docs/``, sorted by filename.

    Returns:
        list[dict]: Each dict has ``name`` (str), ``path`` (Path), ``size_bytes`` (int).
    """
    folder_path = docs_folder_path()
    if not folder_path.is_dir():
        return []
    out = []
    for p in sorted(folder_path.glob("*.pdf")):
        try:
            sz = p.stat().st_size
        except OSError:
            sz = 0
        out.append({"name": p.name, "path": p, "size_bytes": sz})
    return out


def _get_nvidia_api_key():
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        raise ValueError(
            "NVIDIA_API_KEY is required when using NVIDIA NIM deployment."
        )
    return key


def _openai_compatible_base_url():
    """Base URL for OpenAI-compatible servers (vLLM, etc.); must end with /v1."""
    base = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _local_vllm_llm(
    model: str, temperature: float, top_p: float, top_k: int
) -> ChatOpenAI:
    """Chat LLM against a local OpenAI-compatible endpoint (e.g. vLLM serve)."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        top_p=top_p,
        base_url=_openai_compatible_base_url(),
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        max_tokens=1024,
        extra_body={"top_k": top_k},
    )


def _nim_chat_llm(
    model: str, temperature: float, top_p: float, top_k: int
) -> ChatNVIDIA:
    api_key = _get_nvidia_api_key()
    return ChatNVIDIA(
        model=model,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        max_tokens=1024,
        model_kwargs={"top_k": top_k},
    )


def chat_completion(
    deployment: str,
    model_name: str,
    temperature: float,
    top_p: float,
    top_k: int,
    messages: list,
) -> str:
    """Multi-turn chat with no retrieval (plain LLM).

    messages: list of {"role": "user"|"assistant", "content": str}
    """
    mid = get_model(model_name, deployment)
    if not mid:
        raise ValueError("Could not resolve model id for chat.")

    lc_messages = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))

    if deployment in ("Local vLLM", "Local vLLM (OpenAI API)"):
        llm = _local_vllm_llm(mid, temperature, top_p, top_k)
    elif deployment == "NVIDIA NIM":
        llm = _nim_chat_llm(mid, temperature, top_p, top_k)
    else:
        raise ValueError(f"Unsupported deployment for chat: {deployment}")

    response = llm.invoke(lc_messages)
    if hasattr(response, "content"):
        return response.content
    return str(response)


def _embedding_device() -> str:
    """Sentence-transformers device: auto (prefer CUDA when available), cpu, or cuda."""
    raw = os.getenv("ARROW_RAG_EMBEDDINGS_DEVICE", "auto").strip().lower()
    if raw == "cpu":
        return "cpu"
    if raw == "cuda":
        return "cuda"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _embedding_batch_size() -> int:
    try:
        return max(1, int(os.getenv("ARROW_RAG_EMBEDDING_BATCH_SIZE", "8")))
    except ValueError:
        return 8


_EMB: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Lazy singleton — configurable device/batch/model to reduce CUDA OOM."""
    global _EMB
    if _EMB is not None:
        return _EMB
    device = _embedding_device()
    model_name = os.getenv(
        "ARROW_RAG_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    ).strip()
    bs = _embedding_batch_size()
    encode_kw = {"batch_size": bs}
    query_kw = {"batch_size": max(1, min(bs, 64))}
    _EMB = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs=encode_kw,
        query_encode_kwargs=query_kw,
        show_progress=False,
    )
    return _EMB


def get_model(model, deployment):
    """Maps user-friendly model names to their corresponding model identifiers.
    
    Args:
        model (str): The selected model name
        deployment (str): The deployment type (Local vLLM or NVIDIA NIM)
    
    Returns:
        str: The corresponding model identifier for the selected deployment
    """
    if deployment in ("Local vLLM", "Local vLLM (OpenAI API)"):
        # If set, must match the model id passed to `vllm serve --model ...`.
        forced = os.getenv("VLLM_MODEL")
        if forced:
            return forced
        if model == "Meta Llama 3.1":
            return os.getenv(
                "VLLM_MODEL_LLAMA", "meta-llama/Llama-3.1-8B-Instruct"
            )
        if model == "Google Gemma 2":
            return os.getenv("VLLM_MODEL_GEMMA", "google/gemma-2-9b-it")
        if model == "Microsoft Phi 3.5":
            return os.getenv(
                "VLLM_MODEL_PHI", "microsoft/Phi-3.5-mini-instruct"
            )
        if model == "NVIDIA Nemotron 3 Nano":
            return os.getenv(
                "VLLM_MODEL_NEMOTRON",
                "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
            )
    if deployment == "NVIDIA NIM":
        if model == "Meta Llama 3.1":
            return "meta/llama-3.1-8b-instruct"
        if model == "Google Gemma 2":
            return "google/gemma-2-9b-it"
        if model == "Microsoft Phi 3.5":
            return "microsoft/phi-3.5-moe-instruct"
        if model == "NVIDIA Nemotron 3 Nano":
            return os.getenv(
                "NIM_MODEL_NEMOTRON", "nvidia/nemotron-3-nano-30b-a3b"
            )
    
def local_get_answer_upload_pdf(
    model, temperature, top_p, top_k, file_name, query
):
    """Process a single uploaded PDF file using local vLLM (OpenAI API).
    
    Args:
        model (str): The model identifier to use
        temperature (float): The model's temperature setting
        top_p (float): Nucleus sampling threshold
        top_k (int): Top-k token cutoff for the server
        file_name (str): Name of the uploaded PDF file
        query (str): The user's question
    
    Returns:
        str: The model's response to the query
    """

    llm = _local_vllm_llm(model, temperature, top_p, top_k)

    os.write(1,f"{model}\n".encode())
    os.write(1,f"{temperature}\n".encode())

    file_path = f"{working_dir}/{file_name}"
    
    # loading the document
    loader = UnstructuredLoader(file_path)
    documents  = loader.load()
    
    # create text chunks
    
    text_splitter = CharacterTextSplitter(separator="/n",
                                          chunk_size = 1000,
                                          chunk_overlap = 200)
    
    text_chunks = text_splitter.split_documents(documents)
    
    
    # vector embeddings from text chunks 
    
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever = knowledge_base.as_retriever()
        
    )
    
    response = qa_chain.invoke({"query": query})
    
    return response["result"]

def local_get_answer_url(model, temperature, top_p, top_k, query):
    """Process multiple URLs specified here using local vLLM (OpenAI API).
    
    Args:
        model (str): The model identifier to use
        temperature (float): The model's temperature setting
        top_p (float): Nucleus sampling threshold
        top_k (int): Top-k token cutoff for the server
        query (str): The user's question
    
    Returns:
        str: The model's response to the query
    """

    llm = _local_vllm_llm(model, temperature, top_p, top_k)

    os.write(1,f"{model}\n".encode())
    os.write(1,f"{temperature}\n".encode())

    urls = _rag_website_urls()

    os.write(1,f"{urls}\n".encode())
    


    
    # Load documents from the URLs
    docs = [WebBaseLoader(url).load() for url in urls]

    docs_list = [item for sublist in docs for item in sublist]

    
    # create text chunks
    
    text_splitter = CharacterTextSplitter(separator="/n",
                                          chunk_size = 1000,
                                          chunk_overlap = 200)
    
    text_chunks = text_splitter.split_documents(docs_list)
    
    
    # vector embeddings from text chunks 
    
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever = knowledge_base.as_retriever()
    )
    
    response = qa_chain.invoke({"query": query})
    

    return response["result"]

def local_get_answer_folder_pdf(model, temperature, top_p, top_k, query):
    """Process all PDFs in the docs folder using local vLLM (OpenAI API).
    
    Args:
        model (str): The model identifier to use
        temperature (float): The model's temperature setting
        top_p (float): Nucleus sampling threshold
        top_k (int): Top-k token cutoff for the server
        query (str): The user's question
    
    Returns:
        str: The model's response to the query
    """
    
    llm = _local_vllm_llm(model, temperature, top_p, top_k)

    os.write(1,f"{model}\n".encode())
    os.write(1,f"{temperature}\n".encode())

    folder_path = docs_folder_path()

    pdf_files = [file for file in folder_path.glob("*.pdf")]

    # loading the document
    docs = UnstructuredLoader(pdf_files).load()
  
    
    # create text chunks
    
    text_splitter = CharacterTextSplitter(separator="/n",
                                          chunk_size = 1000,
                                          chunk_overlap = 200)
    
    text_chunks = text_splitter.split_documents(docs)
    
    
    # vector embeddings from text chunks 
    
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever = knowledge_base.as_retriever()
        
    )
    
    response = qa_chain.invoke({"query": query})
    
    return response["result"]


def nim_get_answer_folder_pdf(model, temperature, top_p, top_k, query):
    """Process all PDFs in the docs folder using NVIDIA NIM deployment.
    
    Args:
        model (str): The model identifier to use
        temperature (float): The model's temperature setting
        top_p (float): Nucleus sampling threshold
        top_k (int): Top-k token cutoff (passed in model_kwargs when supported)
        query (str): The user's question
    
    Returns:
        str: The model's response to the query
    """

    folder_path = docs_folder_path()
    llm = _nim_chat_llm(model, temperature, top_p, top_k)

    os.write(1,f"{model}\n".encode())
    os.write(1,f"{temperature}\n".encode())

    pdf_files = [file for file in folder_path.glob("*.pdf")]
    docs = UnstructuredLoader(pdf_files).load()

    text_splitter = CharacterTextSplitter(
        separator="/n", chunk_size=1000, chunk_overlap=200
    )
    text_chunks = text_splitter.split_documents(docs)
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=knowledge_base.as_retriever(),
    )
    response = qa_chain.invoke({"query": query})
    return response["result"]

def nim_get_answer_url(model, temperature, top_p, top_k, query):
    """Process multiple URLs specified here using NVIDIA NIM deployment.
    
    Args:
        model (str): The model identifier to use
        temperature (float): The model's temperature setting
        top_p (float): Nucleus sampling threshold
        top_k (int): Top-k token cutoff (passed in model_kwargs when supported)
        query (str): The user's question
    
    Returns:
        str: The model's response to the query
    """
    
    llm = _nim_chat_llm(model, temperature, top_p, top_k)

    os.write(1,f"{model}\n".encode())
    os.write(1,f"{temperature}\n".encode())

    urls = _rag_website_urls()

    # Load documents from the URLs
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # create text chunks
    text_splitter = CharacterTextSplitter(separator="/n",
                                          chunk_size=1000,
                                          chunk_overlap=200)
    
    text_chunks = text_splitter.split_documents(docs_list)

    # vector embeddings from text chunks 
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=knowledge_base.as_retriever()
    )
    
    response = qa_chain.invoke({"query": query})
    
    return response["result"]

def nim_get_answer_upload_pdf(
    model, temperature, top_p, top_k, file_name, query
):
    """Process a single uploaded PDF file using NVIDIA NIM deployment.
    
    Args:
        model (str): The model identifier to use
        temperature (float): The model's temperature setting
        top_p (float): Nucleus sampling threshold
        top_k (int): Top-k token cutoff (passed in model_kwargs when supported)
        file_name (str): Name of the uploaded PDF file
        query (str): The user's question
    
    Returns:
        str: The model's response to the query
    """
    
    llm = _nim_chat_llm(model, temperature, top_p, top_k)

    os.write(1,f"{model}\n".encode())
    os.write(1,f"{temperature}\n".encode())

    file_path = f"{working_dir}/{file_name}"
    
    # loading the document
    loader = UnstructuredLoader(file_path)
    documents = loader.load()
    
    # create text chunks
    text_splitter = CharacterTextSplitter(separator="/n",
                                          chunk_size=1000,
                                          chunk_overlap=200)
    
    text_chunks = text_splitter.split_documents(documents)

    # vector embeddings from text chunks 
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=knowledge_base.as_retriever()
    )
    
    response = qa_chain.invoke({"query": query})
    
    return response["result"]