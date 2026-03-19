from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm(provider: str, **kwargs):
    provider = provider.lower()

    if provider == "ollama":
        return OllamaLLM(
            model=kwargs.get("model", "dolphin-mistral"),
            base_url=kwargs.get("base_url", "http://localhost:11434"),
            temperature=kwargs.get("temperature", 0.3),
        )

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=kwargs.get("model", "gemini-1.5-pro"),
            temperature=kwargs.get("temperature", 0.3),
            google_api_key=kwargs["api_key"],
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
