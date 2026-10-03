"""
Baseline 0 - Socia's production heuristic, reproduced faithfully.

Real behavior (see synthesize_patterns_from_observations): once any tag
reaches 3+ occurrences for a user, it is considered a detected pattern
from that point forward - confidence only ever strengthens, it is never
revoked. We reproduce that: once a tag crosses the count-3 threshold at
the timestamp of its 3rd occurrence, the detection remains "active" for
the rest of that user's timeline.

This baseline does not consider order at all - it is a pure per-tag
frequency count, which is expected to make it prone to false positives
(e.g. flagging reordered hard negatives, since it only cares that the
same tag occurred 3+ times somewhere, not in what order).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

MIN_COUNT = 3


def detect_heuristic_instances(
    user_events: List[Tuple[datetime, str]],
) -> List[Tuple[datetime, datetime, str]]:
    """
    user_events: chronologically sorted list of (created_at, event_type)
    for a single user.

    Returns a list of (span_start, span_end, tag) detected instances.
    span_start = timestamp of the tag's 3rd occurrence.
    span_end   = timestamp of the user's last event overall (detection
                 stays active for the rest of the timeline, matching
                 production behavior).
    """

    if not user_events:
        return []

    last_event_time = user_events[-1][0]

    occurrences_by_tag: Dict[str, List[datetime]] = defaultdict(list)

    for timestamp, tag in user_events:
        occurrences_by_tag[tag].append(timestamp)

    instances = []

    for tag, timestamps in occurrences_by_tag.items():

        if len(timestamps) >= MIN_COUNT:

            third_occurrence_time = timestamps[MIN_COUNT - 1]

            instances.append(
                (third_occurrence_time, last_event_time, tag)
            )

    return instances