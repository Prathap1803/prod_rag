from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from deva.config import (
    LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE,
    OLLAMA_BASE_URL, GEMINI_API_KEY,
)
from deva.logger import get_logger

logger = get_logger(__name__)


def get_llm(provider: str = LLM_PROVIDER, model: str = LLM_MODEL, **kwargs):
    provider = provider.lower()
    temperature = kwargs.get("temperature", LLM_TEMPERATURE)
    logger.info(f"Loading LLM | provider={provider} | model={model}")

    if provider == "ollama":
        return OllamaLLM(
            model=model,
            base_url=kwargs.get("base_url", OLLAMA_BASE_URL),
            temperature=temperature,
        )

    if provider == "gemini":
        api_key = kwargs.get("api_key") or GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini provider")
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
