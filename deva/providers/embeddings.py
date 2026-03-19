from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_embeddings(provider: str, **kwargs):
    provider = provider.lower()

    if provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=kwargs.get(
                "model", "sentence-transformers/all-MiniLM-L6-v2"
            )
        )

    if provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=kwargs["api_key"],
        )

    raise ValueError(f"Unsupported embeddings provider: {provider}")
