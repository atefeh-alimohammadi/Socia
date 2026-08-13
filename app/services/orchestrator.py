import logging

logger = logging.getLogger(__name__)


SHORT_TRANSACTIONAL_MESSAGES = {
    "ok",
    "okay",
    "yes",
    "no",
    "yeah",
    "yep",
    "nope",
    "sure",
    "thanks",
    "thank you",
    "thx",
    "got it",
    "understood",
}


def route_message(content: str) -> dict:
    normalized_content = content.strip().lower()

    if normalized_content in SHORT_TRANSACTIONAL_MESSAGES:
        result = {
            "needs_episodic_retrieval": False,
            "needs_full_context": False,
        }
    else:
        result = {
            "needs_episodic_retrieval": True,
            "needs_full_context": True,
        }

    logger.warning("ORCHESTRATOR ROUTE: %s", result)

    return result