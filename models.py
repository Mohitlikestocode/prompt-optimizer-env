from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from openenv.core.env_server import Action, Observation, State


class PromptOptimizerAction(Action):
    """
    The agent's action each step: a rewritten prompt string.

    The agent receives the current task spec and last feedback,
    then rewrites the prompt to try to satisfy the target output spec.
    """
    rewritten_prompt: str  # The new prompt the agent wants to test


class PromptOptimizerObservation(Observation):
    """
    What the agent sees after each step.

    Contains rich feedback so the agent understands WHY it failed
    and can make a targeted improvement next step.
    """
    # The task specification (stays constant across steps)
    task_description: str         # e.g., "Make the LLM output JSON with keys: name, age, city"
    target_spec_type: str         # "json_keys" | "code_tests" | "phrase_match" | "length_range" | "refusal"
    target_spec_value: Any        # The actual target (list of keys, test code, phrase string, range tuple)

    # The current prompt being tested (what was sent to judge)
    current_prompt: str

    # Judge's response to the current prompt
    judge_response: str

    # Structured feedback on what passed and what failed
    checks_passed: List[str]      # e.g., ["has_json_structure", "has_name_key"]
    checks_failed: List[str]      # e.g., ["missing_age_key", "missing_city_key"]

    # Reward for this step
    step_reward: float            # between 0.0 and 1.0

    # Episode state
    step_number: int
    done: bool = False
    success: bool = False         # True if all checks passed (score == 1.0)

    # The history of prompts tried so far (last 3, for context)
    prompt_history: List[str]
    reward_history: List[float]


class PromptOptimizerState(State):
    """Internal environment state."""
    episode_id: str
    step_count: int
    task_id: str
    best_score_so_far: float
    current_prompt: str
    task_description: str
    target_spec_type: str
    target_spec_value: Any
    initial_bad_prompt: str
    prompt_history: List[str]
    reward_history: List[float]
