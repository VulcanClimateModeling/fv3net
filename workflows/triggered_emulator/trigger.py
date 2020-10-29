def is_active(a):
    threshold_tendency = 0.01 / 86400  # K/day
    return ((a.dQ1 ** 2 + (2.51e6 / 1004 * a.dQ2) ** 2) > threshold_tendency ** 2).any(
        "z"
    )
