from deva.logger import get_logger

logger = get_logger(__name__)

BLOCKLIST = [
    "kill yourself",
    "how to make a bomb",
    "credit card number",
    "social security number",
]

UNHELPFUL_RESPONSES = {
    "i don't have enough information.",
    "i don't know.",
    "",
}


def _is_safe(text: str) -> bool:
    lowered = text.lower()
    return not any(b in lowered for b in BLOCKLIST)


def _is_helpful(text: str) -> bool:
    return text.strip().lower() not in UNHELPFUL_RESPONSES


def enforce(answer: str) -> str:
    if not answer or not answer.strip():
        logger.warning("Empty answer returned by LLM")
        return "I don't have enough information to answer that."

    if not _is_safe(answer):
        logger.warning("Guardrail triggered: unsafe content in answer")
        return "I can't answer that safely."

    if not _is_helpful(answer):
        logger.info("Answer flagged as unhelpful — returning as-is")

    return answer


def validate_input(question: str) -> tuple[bool, str]:
    if not question or not question.strip():
        return False, "Question cannot be empty."
    if len(question) > 1000:
        return False, "Question too long (max 1000 characters)."
    if not _is_safe(question):
        return False, "Input contains disallowed content."
    return True, ""
