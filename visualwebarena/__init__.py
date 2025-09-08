"""
VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks

This package provides components for evaluating multimodal autonomous language agents
on web-based visual tasks.
"""

__version__ = "0.1.0"

from .browser_env import ScriptBrowserEnv
from .agent import PromptAgent

__all__ = ["ScriptBrowserEnv", "PromptAgent"]
