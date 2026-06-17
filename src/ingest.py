import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig
from src import config


class StellaEmbeddings(Embeddings):
    """
    Custom LangChain-compatible embeddings class that loads Stella directly
    via SentenceTransformer, patching the config at runtime to disable
    xformers-dependent memory_efficient_attention (CPU-safe).
    """

    def __init__(self, model_name: str, trust_remote_code: bool = True):
        # Patch the Stella config before the model is loaded so that the
        # attention implementation doesn't require xformers
        stella_config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        if hasattr(stella_config, "use_memory_efficient_attention"):
            stella_config.use_memory_efficient_attention = False
        if hasattr(stella_config, "unpad_inputs"):
            stella_config.unpad_inputs = False

        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=trust_remote_code,
            config_kwargs={
                "use_memory_efficient_attention": False,
                "unpad_inputs": False,
            }
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document chunks — no instruction prompt."""
        processed = [t.replace("\n", " ") for t in texts]
        embeddings = self.model.encode(processed, normalize_embeddings=True)
        return [list(map(float, e)) for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query using the Stella s2p_query instruction."""
        processed = text.replace("\n", " ")
        embedding = self.model.encode(
            processed,
            prompt_name=config.EMBEDDING_PROMPT_NAME,
            normalize_embeddings=True
        )
        return list(map(float, embedding))


def load_document(file_path: str):
    """
    Loads document based on file extension.
    Supports PDF and TXT.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {ext}. Please upload a PDF or TXT file.")

    return loader.load()


def chunk_documents(documents):
    """
    Splits documents into smaller overlapping chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


def get_embedding_model():
    """
    Initializes the Stella embedding model (CPU-safe, no xformers required).
    """
    return StellaEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        trust_remote_code=config.EMBEDDING_TRUST_REMOTE_CODE,
    )


def create_vector_store(chunks, embedding_model):
    """
    Creates an in-memory FAISS vector store using cosine similarity.
    Since embeddings are L2-normalized, inner product == cosine similarity,
    producing scores in [0, 1] range.
    """
    return FAISS.from_documents(
        chunks,
        embedding_model,
        distance_strategy=DistanceStrategy.COSINE
    )

