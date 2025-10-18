from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import torch

from settings import get_settings, get_logger

logger = get_logger(__name__)

def create_chunks(text: str) -> list[str]:
    settings = get_settings()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
    )
    if not text.strip():
        return []

    return text_splitter.split_text(text)


def create_embedding_model() -> SentenceTransformer:
    settings = get_settings()
    gpu_count: int = torch.cuda.device_count()
    gpu_id: int = settings.EMBEDDINGS_GPU_ID
    device: str = "cpu"
    if torch.cuda.is_available():
        if not (0 <= gpu_id < gpu_count):
            raise ValueError(f"Seu animal. ID: {gpu_id} invalido")
        device = f"cuda:{gpu_id}"
    logger.info(f"Device: {device}")
    try:
        model = SentenceTransformer(
            settings.EMBEDDING_MODEL,
            trust_remote_code=True,
            token=settings.HF_TOKEN.get_secret_value(),
        )
        model.to(device)
        return model
    except Exception as e:
        raise Exception(
            f"Falha ao carregar o modelo '{settings.EMBEDDING_MODEL}': {e}"
        )
        
def create_embeddings(
    texts: list[str], model: SentenceTransformer
) -> list[list[float]]:
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()
