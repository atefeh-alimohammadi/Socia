import logging
import re

logger = logging.getLogger(__name__)


def check_message_safety(content: str) -> dict:
    logger.warning("SAFETY CHECK RUN: %s", content)
    text = content.lower().strip()

    high_risk_patterns = {
        "self_harm": [
            r"\bkill myself\b",
            r"\bend my life\b",
            r"\btake my own life\b",
            r"\bhurt myself\b",
            r"\bself[- ]harm\b",
            r"\bsuicide\b",
            r"\bsuicidal\b",
            r"\bwant to die\b",
            r"\bwish i was dead\b",
            r"\bwish i were dead\b",
        ],
    }

    matched_signals = []

    for signal_type, patterns in high_risk_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text):
                matched_signals.append(signal_type)
                break

    if matched_signals:
        logger.warning(
            "Safety event triggered: risk_level=high signals=%s",
            matched_signals,
        )

        return {
            "risk_level": "high",
            "signals": matched_signals,
        }

    return {
        "risk_level": "none",
        "signals": [],
    }