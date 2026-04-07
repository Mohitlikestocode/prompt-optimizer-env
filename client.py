from __future__ import annotations
from typing import Any, Dict
from openenv.core.env_client import EnvClient, StepResult
from models import PromptOptimizerAction, PromptOptimizerObservation, PromptOptimizerState


class PromptOptimizerEnv(EnvClient[PromptOptimizerAction, PromptOptimizerObservation, PromptOptimizerState]):

    def _step_payload(self, action: PromptOptimizerAction) -> Dict[str, Any]:
        return {"rewritten_prompt": action.rewritten_prompt}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[PromptOptimizerObservation]:
        obs_data = payload.get("observation", payload)
        # done lives at the result level in openenv — inject it back into obs
        if "done" not in obs_data:
            obs_data["done"] = payload.get("done", False)
        obs = PromptOptimizerObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", obs.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> PromptOptimizerState:
        return PromptOptimizerState(**payload)
