# ============================================================
# CRITICAL TIMESTEP SELECTION
# ============================================================

def detect_critical_timesteps(episode_log,
                              queue_jump_threshold=5):
    """
    episode_log: list of dicts with keys:
        queue_len_NS, queue_len_EW
        emergency_count_NS, emergency_count_EW
        phase
    """
    critical_indices = []

    for t in range(1, len(episode_log)):
        prev = episode_log[t - 1]
        curr = episode_log[t]

        emergency_event = (
            curr["emergency_count_NS"] > prev["emergency_count_NS"] or
            curr["emergency_count_EW"] > prev["emergency_count_EW"]
        )

        queue_jump = (
            abs(curr["queue_len_NS"] - prev["queue_len_NS"]) >= queue_jump_threshold or
            abs(curr["queue_len_EW"] - prev["queue_len_EW"]) >= queue_jump_threshold
        )

        phase_change = curr["phase"] != prev["phase"]

        if emergency_event or queue_jump or phase_change:
            critical_indices.append(t)

    return critical_indices
