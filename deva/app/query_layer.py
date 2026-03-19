from deva.logger import get_logger

logger = get_logger(__name__)


def classify_intent(question: str) -> Intent:
    logger.debug(f"Classifying intent for: {question!r}")
    llm = _get_llm()
    out = (INTENT_PROMPT | llm).invoke({"question": question})
    label = str(out).strip().lower()
    intent = Intent(label) if label in {i.value for i in Intent} else Intent.QA
    logger.info(f"Intent classified: {intent.value}")
    return intent


def enhance_query(question: str, intent: Intent, context_hint: str = "") -> str:
    logger.debug(f"Enhancing query | intent={intent.value} | question={question!r}")
    llm = _get_llm()
    out = (ENHANCE_PROMPT | llm).invoke({
        "intent": intent.value,
        "question": question,
        "context_hint": context_hint,
    })
    enhanced = str(out).strip()
    logger.info(f"Enhanced query: {enhanced!r}")
    return enhanced
