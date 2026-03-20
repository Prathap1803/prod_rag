from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from deva.config import EMBEDDINGS_PROVIDER, EMBEDDINGS_MODEL, GEMINI_API_KEY
from deva.logger import get_logger

logger = get_logger(__name__)


def get_embeddings(
    provider: str = EMBEDDINGS_PROVIDER,
    model: str = EMBEDDINGS_MODEL,
    **kwargs,
):
    provider = provider.lower()
    logger.info(f"Loading embeddings | provider={provider} | model={model}")

    if provider == "huggingface":
        return HuggingFaceEmbeddings(model_name=model)

    if provider == "gemini":
        api_key = kwargs.get("api_key") or GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini embeddings")
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )

    raise ValueError(f"Unsupported embeddings provider: {provider}")
