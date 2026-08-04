"""Rate-limited, monotonic trajectory reference progress."""


def advance_reference_progress(s_ref, s_star, step=0.01):
    """Advance a reference no faster than ``step`` and never move backward."""
    if s_ref is None:
        return max(0.0, min(1.0, s_star))
    target = max(s_ref, s_star)
    return min(1.0, s_ref + step, target)
