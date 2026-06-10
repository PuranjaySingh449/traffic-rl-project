import numpy as np
from rl.env import TrafficSignalEnv


class TrafficSignalEnvPPO(TrafficSignalEnv):
    """
    PPO-specific environment with stabilized reward shaping.
    Uses relative improvement instead of absolute penalties.
    """

    def __init__(self, auto_close=False, scenario="baseline", seed=0):
        self.seed = seed
        super().__init__(
            auto_close=auto_close,
            scenario=scenario,
            seed=seed
        )

        self.prev_emergency_wait = None
        self.prev_queue = None

    def reset(self):
        state = super().reset()

        # Initialize PPO memory
        self.prev_emergency_wait = self._get_emergency_wait()
        self.prev_queue = self._get_queue_length()

        return state

    # -----------------------------
    # PPO Reward Function (FINAL)
    # -----------------------------
    def _compute_reward(self, prev_phase=None):

        emergency_wait = self._get_emergency_wait()
        queue_len = self._get_queue_length()

        emergency_improvement = (
            self.prev_emergency_wait - emergency_wait
            if self.prev_emergency_wait is not None else 0.0
        )

        queue_improvement = (
            self.prev_queue - queue_len
            if self.prev_queue is not None else 0.0
        )

        phase_switch = 1.0 if prev_phase != self.last_phase else 0.0

        reward = 0.0

        if emergency_wait > 0:
            reward += 2.0 * emergency_improvement

        reward += 0.5 * queue_improvement
        reward -= 0.02 * queue_len

        reward -= 1.5 * phase_switch

        reward = np.clip(reward, -10.0, 5.0)

        self.prev_emergency_wait = emergency_wait
        self.prev_queue = queue_len

        return reward
