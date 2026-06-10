# ============================================================
# XAI FEATURE CONFIG (7-D STATE – FINAL)
# ============================================================

FEATURE_NAMES = [
    "queue_ns",              # 0
    "queue_ew",              # 1
    "waiting_time_ns",       # 2
    "waiting_time_ew",       # 3
    "emergency_present",     # 4
    "emergency_distance",    # 5
    "current_phase"          # 6
]

FEATURE_GROUPS = {
    "Queue Length": [0, 1],
    "Waiting Time": [2, 3],
    "Emergency": [4, 5],
    "Signal State": [6],
}
