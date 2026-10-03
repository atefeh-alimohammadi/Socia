"""
Baseline 1 - simplified WINEPI-style frequent serial episode mining,
per user (Mannila, Toivonen & Verkamo, 1997).

Scope decision: mining is done PER USER, not pooled across the cohort.
This matches the actual problem definition (a pattern must recur for
THIS user, not be globally frequent across all simulated users) - a
corpus-wide frequency definition would silently answer a different,
mismatched question.

min_support = 3, matching Baseline 0's own threshold exactly, so both
baselines are held to the same recurrence bar rather than min_support
being separately tuned for WINEPI.

Window size = W_PATTERN_HOURS = 72 hours, fixed from the data generator,
not tuned on validation or test data, per agreed methodology.

Serial (ordered) episodes only, lengths 2-4, matching the pattern
lengths supported by the synthetic dataset. Apriori-style level-wise
candidate generation: a k-episode is only considered as a candidate if
its (k-1)-length prefix and suffix subepisodes are already frequent.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

MIN_SUPPORT = 3
WINDOW_HOURS = 72
MAX_EPISODE_LENGTH = 4


def _windows_containing_subsequence(
    anchor_windows: List[List[str]],
    episode: Tuple[str, ...],
) -> List[int]:
    """
    Returns the indices of anchor_windows in which `episode` (an ordered
    tuple of categories) appears as a subsequence (not necessarily
    contiguous) of that window's chronological category list.
    """

    matching_indices = []

    for idx, window_categories in enumerate(anchor_windows):

        pos = 0

        for cat in window_categories:

            if pos < len(episode) and cat == episode[pos]:
                pos += 1

            if pos == len(episode):
                matching_indices.append(idx)
                break

    return matching_indices


def detect_winepi_instances(
    user_events: List[Tuple[datetime, str]],
) -> List[Tuple[datetime, datetime, Tuple[str, ...]]]:
    """
    user_events: chronologically sorted list of (created_at, event_type)
    for a single user.

    Returns a list of (span_start, span_end, episode) detected instances -
    one per anchor-window occurrence of every episode found frequent
    (support >= MIN_SUPPORT) for this user.
    """

    if len(user_events) < 2:
        return []

    window_delta = timedelta(hours=WINDOW_HOURS)

    # Anchor a window at every event's timestamp (standard WINEPI
    # construction), collecting the ordered categories of all events
    # falling within [anchor_time, anchor_time + window_delta).
    anchor_starts: List[datetime] = []
    anchor_windows: List[List[str]] = []

    n = len(user_events)

    for i in range(n):

        anchor_time = user_events[i][0]
        window_end = anchor_time + window_delta

        categories_in_window = []

        j = i
        while j < n and user_events[j][0] < window_end:
            categories_in_window.append(user_events[j][1])
            j += 1

        anchor_starts.append(anchor_time)
        anchor_windows.append(categories_in_window)

    # --- Level 1: frequent single categories (support only used for
    # Apriori pruning of level-2 candidates, not reported as detections) ---
    support_1: Dict[str, int] = defaultdict(int)

    for window_categories in anchor_windows:
        for cat in set(window_categories):
            support_1[cat] += 1

    frequent_1 = {
        cat for cat, support in support_1.items() if support >= MIN_SUPPORT
    }

    if not frequent_1:
        return []

    all_instances: List[Tuple[datetime, datetime, Tuple[str, ...]]] = []

    # --- Level 2 ---
    level_k_frequent: Dict[Tuple[str, ...], List[int]] = {}

    for a in frequent_1:
        for b in frequent_1:

            if a == b:
                continue

            episode = (a, b)
            matches = _windows_containing_subsequence(anchor_windows, episode)

            if len(matches) >= MIN_SUPPORT:
                level_k_frequent[episode] = matches

    _record_instances(level_k_frequent, anchor_starts, window_delta, all_instances)

    # --- Levels 3 and 4: Apriori-style extension ---
    for k in range(3, MAX_EPISODE_LENGTH + 1):

        candidates = _generate_candidates(level_k_frequent, k)

        next_level_frequent: Dict[Tuple[str, ...], List[int]] = {}

        for episode in candidates:

            matches = _windows_containing_subsequence(anchor_windows, episode)

            if len(matches) >= MIN_SUPPORT:
                next_level_frequent[episode] = matches

        if not next_level_frequent:
            break

        _record_instances(
            next_level_frequent, anchor_starts, window_delta, all_instances
        )

        level_k_frequent = next_level_frequent

    return all_instances


def _generate_candidates(
    frequent_prev: Dict[Tuple[str, ...], List[int]],
    k: int,
) -> List[Tuple[str, ...]]:
    """
    Apriori-style join: a k-episode is a candidate only if both its
    length-(k-1) prefix and its length-(k-1) suffix are already frequent.
    """

    prev_episodes = set(frequent_prev.keys())
    candidates = set()

    for episode_a in prev_episodes:
        for episode_b in prev_episodes:

            if episode_a[1:] == episode_b[:-1]:

                candidate = episode_a + (episode_b[-1],)

                if len(candidate) == k:
                    candidates.add(candidate)

    return list(candidates)


def _record_instances(
    frequent_episodes: Dict[Tuple[str, ...], List[int]],
    anchor_starts: List[datetime],
    window_delta: timedelta,
    all_instances: List[Tuple[datetime, datetime, Tuple[str, ...]]],
) -> None:

    for episode, matching_indices in frequent_episodes.items():
        for idx in matching_indices:

            span_start = anchor_starts[idx]
            span_end = span_start + window_delta

            all_instances.append((span_start, span_end, episode))