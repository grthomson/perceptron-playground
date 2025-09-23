import pandas as pd


def _prep_str(x) -> str:
    """Convert to lowercase string, strip whitespace, handle NaN."""
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def levenshtein_distance(a: str, b: str) -> int:
    """Raw Levenshtein edit distance."""
    a, b = _prep_str(a), _prep_str(b)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalized_levenshtein(a: str, b: str) -> float:
    """Levenshtein similarity normalised to [0,1]."""
    a, b = _prep_str(a), _prep_str(b)
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(a, b)
    return 1.0 - (dist / max_len)
