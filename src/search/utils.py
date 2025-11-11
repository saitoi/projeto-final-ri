from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

from settings import get_settings, get_logger, EMBEDDING_MODELS

logger = get_logger(__name__)

def create_chunks(text: str) -> list[str]:
    settings = get_settings()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    if not text.strip():
        return []

    return text_splitter.split_text(text)


def create_embedding_model(variant: str | None = None) -> SentenceTransformer:
    settings = get_settings()
    gpu_count: int = torch.cuda.device_count()
    gpu_id: int = settings.embedding_gpu_id
    device: str = "cpu"
    if torch.cuda.is_available():
        if not (0 <= gpu_id < gpu_count):
            raise ValueError(
                f"Invalid GPU id '{gpu_id}'. Available CUDA devices: {gpu_count}"
            )
        device = f"cuda:{gpu_id}"
    logger.info(f"Device: {device}")
    embedding_variant = variant if variant is not None else settings.embedding_variant
    model_name = EMBEDDING_MODELS[embedding_variant]

    try:
        model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            token=settings.hf_token.get_secret_value() if settings.hf_token else None,
        )
        model.to(device)
        return model
    except Exception as e:
        raise Exception(
            f"Falha ao carregar o modelo '{model_name}': {e}"
        )

def create_embeddings(
    texts: list[str], model: SentenceTransformer
) -> list[list[float]]:
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()

def _embed_text(text: str, model: SentenceTransformer) -> list[float]:
    chunks = create_chunks(text)
    if len(chunks) == 0:
        return []
    embeddings = create_embeddings(chunks, model)
    return np.mean(embeddings, axis=0).tolist()
