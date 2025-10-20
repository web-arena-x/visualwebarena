from typing import Any, Optional
from browser_env import Trajectory
from browser_env.actions import Action

class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        pass

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        raise NotImplementedError

