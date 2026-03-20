import os
from deva.logger import get_logger

logger = get_logger(__name__)

_langfuse_handler = None


def get_langfuse_handler():
    global _langfuse_handler

    if _langfuse_handler is not None:
        return _langfuse_handler

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.warning(
            "Langfuse disabled — LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set."
        )
        _langfuse_handler = None
        return None

    try:
        from langfuse.langchain import CallbackHandler
        from langfuse import Langfuse

        # Initialize Langfuse client (credentials read from env)
        lf = Langfuse()
        lf.auth_check()  # optional, verifies connection

        # Initialize callback handler (no keys in constructor)
        _langfuse_handler = CallbackHandler()

        logger.info(f"Langfuse tracing enabled → {host}")

    except Exception as e:
        logger.warning(f"Langfuse init failed ({e}) — tracing disabled")
        _langfuse_handler = None

    return _langfuse_handler


def get_callbacks() -> list:
    handler = get_langfuse_handler()
    return [handler] if handler else []