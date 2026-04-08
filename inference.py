#!/usr/bin/env python3
"""
Inference Script — Prompt Optimizer RL Environment
===================================================

MANDATORY STDOUT FORMAT:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

All rewards formatted to 2 decimal places.
done and success are lowercase booleans.
Score is in [0, 1].
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import AsyncOpenAI
from client import PromptOptimizerEnv
from models import PromptOptimizerAction

# ── Configuration ──────────────────────────────────────────────────────────────
IMAGE_NAME   = os.getenv("LOCAL_IMAGE_NAME")
API_KEY      = os.environ.get("API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME    = os.getenv("PROMPT_OPT_TASK", "json_user_profile")
BENCHMARK    = "prompt_optimizer"
MAX_STEPS    = 8
TEMPERATURE  = 0.7
MAX_TOKENS   = 800


# ── Logging ────────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    action_safe = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_safe!r} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Agent System Prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert prompt engineer. Your job is to iteratively rewrite a system prompt
so that a target LLM produces a specific, structured output that meets a given specification.

You will receive:
- task_description: What the output must look like
- target_spec_type: The type of check being applied (json_keys, code_tests, phrase_match, length_range, refusal)
- current_prompt: The prompt that was just tested
- judge_response: What the judge LLM produced with the current prompt
- checks_passed: Which checks are already satisfied
- checks_failed: Which checks are still failing
- prompt_history: Previous prompts tried

Your job: Output ONLY the new rewritten system prompt. Nothing else.
No explanation. No preamble. No "Here is the prompt:". Just the prompt text.

Strategy:
- Read checks_failed carefully — these are the exact things you need to fix
- Look at judge_response to understand what the LLM is producing
- Make targeted, specific improvements each step
- Add format instructions, examples, and constraints as needed
- For JSON: specify exact keys and types in the prompt
- For code: specify exact function signature and test cases
- For phrase match: quote the exact phrase that must appear
- For length: specify exact character limits
- For refusal: instruct the LLM to decline and include required signals
""").strip()


def build_user_prompt(obs_dict: dict) -> str:
    """Build the user message for the agent from the current observation."""
    checks_passed = obs_dict.get("checks_passed", [])
    checks_failed = obs_dict.get("checks_failed", [])
    prompt_history = obs_dict.get("prompt_history", [])
    reward_history = obs_dict.get("reward_history", [])

    history_text = ""
    for i, (p, r) in enumerate(zip(prompt_history[-3:], reward_history[-3:]), 1):
        history_text += f"\nAttempt {i} (score={r:.2f}):\n{p[:300]}\n"

    return textwrap.dedent(f"""
TASK DESCRIPTION:
{obs_dict.get('task_description', '')}

SPEC TYPE: {obs_dict.get('target_spec_type', '')}

CURRENT PROMPT TESTED:
{obs_dict.get('current_prompt', '')[:500]}

JUDGE RESPONSE TO CURRENT PROMPT:
{obs_dict.get('judge_response', '')[:600]}

CHECKS PASSED: {', '.join(checks_passed) if checks_passed else 'None yet'}
CHECKS FAILED: {', '.join(checks_failed) if checks_failed else 'None'}
CURRENT SCORE: {obs_dict.get('step_reward', 0):.2f}

PROMPT HISTORY (last 3 attempts):
{history_text if history_text else 'No history yet.'}

Now write a better prompt that fixes the failing checks. Output ONLY the prompt text.
""").strip()


async def get_agent_action(client: AsyncOpenAI, obs_dict: dict) -> str:
    """Call the agent LLM to get the next rewritten prompt."""
    user_prompt = build_user_prompt(obs_dict)
    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "You are a helpful assistant. Follow the task description exactly."
    except Exception as exc:
        print(f"[DEBUG] Agent LLM call failed: {exc}", flush=True)
        return "You are a helpful assistant. Please follow the task specification carefully."


async def main() -> None:
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if IMAGE_NAME:
        env = await PromptOptimizerEnv.from_docker_image(IMAGE_NAME)
    else:
        base_url = os.getenv("ENV_BASE_URL", "http://localhost:8000")
        env = PromptOptimizerEnv(base_url=base_url)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with env:
            result = await env.reset()
            obs = result.observation

            log_step(
                step=0,
                action=obs.current_prompt,
                reward=obs.step_reward,
                done=obs.done,
                error=None,
            )
            rewards.append(obs.step_reward)

            for step in range(1, MAX_STEPS + 1):
                if obs.done:
                    break

                obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__

                new_prompt = await get_agent_action(client, obs_dict)

                result = await env.step(PromptOptimizerAction(rewritten_prompt=new_prompt))
                obs = result.observation

                rewards.append(obs.step_reward)
                steps_taken = step

                log_step(
                    step=step,
                    action=new_prompt,
                    reward=obs.step_reward,
                    done=obs.done,
                    error=None,
                )

                if obs.done:
                    success = obs.success
                    break

        score = max(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= 1.0

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
