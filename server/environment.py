from __future__ import annotations
import os
import json
import re
import uuid
import random
from typing import Any, Dict, List, Optional, Tuple
import httpx
from openenv.core.env_server import Environment
from models import (
    PromptOptimizerAction,
    PromptOptimizerObservation,
    PromptOptimizerState,
)

# ─────────────────────────────────────────────
# TASK BANK
# Each task = (description, spec_type, spec_value, initial_bad_prompt)
# spec_type determines which Python verifier runs.
# All rewards are fully deterministic — no LLM judge needed for scoring.
# ─────────────────────────────────────────────

TASK_BANK = [
    # ── JSON STRUCTURE TASKS ──────────────────────────────────────────
    {
        "task_id": "json_user_profile",
        "description": (
            "Make the LLM output a valid JSON object with exactly these keys: "
            "'name' (string), 'age' (integer), 'city' (string), 'email' (string). "
            "No extra keys. No prose before or after the JSON."
        ),
        "spec_type": "json_keys",
        "spec_value": {
            "required_keys": ["name", "age", "city", "email"],
            "type_checks": {"age": "int"},
        },
        "initial_bad_prompt": "Tell me about a person.",
    },
    {
        "task_id": "json_product_listing",
        "description": (
            "Make the LLM output a valid JSON object with keys: "
            "'product_name', 'price' (float), 'in_stock' (boolean), 'category' (string). "
            "Values must be realistic. No surrounding text."
        ),
        "spec_type": "json_keys",
        "spec_value": {
            "required_keys": ["product_name", "price", "in_stock", "category"],
            "type_checks": {"price": "float", "in_stock": "bool"},
        },
        "initial_bad_prompt": "Describe a product for me.",
    },
    # ── CODE TEST TASKS ───────────────────────────────────────────────
    {
        "task_id": "code_fibonacci",
        "description": (
            "Make the LLM produce a Python function called 'fibonacci(n)' that returns "
            "the nth Fibonacci number (0-indexed). It must pass: "
            "fibonacci(0)==0, fibonacci(1)==1, fibonacci(7)==13, fibonacci(10)==55."
        ),
        "spec_type": "code_tests",
        "spec_value": {
            "test_cases": [
                ("fibonacci(0)", 0),
                ("fibonacci(1)", 1),
                ("fibonacci(7)", 13),
                ("fibonacci(10)", 55),
            ],
            "function_name": "fibonacci",
        },
        "initial_bad_prompt": "Write some code.",
    },
    {
        "task_id": "code_palindrome",
        "description": (
            "Make the LLM produce a Python function called 'is_palindrome(s)' that returns "
            "True if s is a palindrome (case-insensitive), False otherwise. "
            "Must pass: is_palindrome('racecar')==True, is_palindrome('hello')==False, "
            "is_palindrome('Madam')==True."
        ),
        "spec_type": "code_tests",
        "spec_value": {
            "test_cases": [
                ("is_palindrome('racecar')", True),
                ("is_palindrome('hello')", False),
                ("is_palindrome('Madam')", True),
                ("is_palindrome('A man')", False),
            ],
            "function_name": "is_palindrome",
        },
        "initial_bad_prompt": "Write something about strings.",
    },
    # ── PHRASE MATCH TASKS ────────────────────────────────────────────
    {
        "task_id": "phrase_disclaimer",
        "description": (
            "Make the LLM include the EXACT phrase 'This is not financial advice.' "
            "somewhere in its response to a question about investing."
        ),
        "spec_type": "phrase_match",
        "spec_value": {
            "required_phrases": ["This is not financial advice."],
            "context_prompt_for_judge": "What stocks should I buy right now?",
        },
        "initial_bad_prompt": "What stocks should I buy?",
    },
    {
        "task_id": "phrase_step_format",
        "description": (
            "Make the LLM respond to 'How do I make pasta?' using numbered steps "
            "('1.', '2.', '3.') and include at least 4 steps."
        ),
        "spec_type": "phrase_match",
        "spec_value": {
            "required_phrases": ["1.", "2.", "3.", "4."],
            "context_prompt_for_judge": "How do I make pasta?",
        },
        "initial_bad_prompt": "Talk about pasta.",
    },
    # ── LENGTH RANGE TASKS ────────────────────────────────────────────
    {
        "task_id": "length_tweet",
        "description": (
            "Make the LLM summarize 'climate change' in exactly 280 characters or fewer, "
            "but at least 100 characters. No hashtags."
        ),
        "spec_type": "length_range",
        "spec_value": {
            "min_chars": 100,
            "max_chars": 280,
            "forbidden_patterns": ["#"],
            "context_prompt_for_judge": "Summarize climate change.",
        },
        "initial_bad_prompt": "Write something about climate.",
    },
    # ── REFUSAL TASKS ─────────────────────────────────────────────────
    {
        "task_id": "refusal_medical",
        "description": (
            "Make the LLM politely refuse to give specific medical dosage advice "
            "when asked 'What dose of ibuprofen should I take for severe pain?' "
            "The response must contain 'doctor' or 'healthcare professional' and "
            "must NOT contain any specific mg amounts."
        ),
        "spec_type": "refusal",
        "spec_value": {
            "required_phrases_any": ["doctor", "healthcare professional", "medical professional"],
            "forbidden_patterns": [r"\d+\s*mg", r"\d+\s*milligrams"],
            "context_prompt_for_judge": "What dose of ibuprofen should I take for severe pain?",
        },
        "initial_bad_prompt": "Answer this medical question directly.",
    },
]

MAX_STEPS = 8
MAX_JUDGE_TOKENS = 512


def _call_judge(system_prompt: str, user_prompt: str) -> str:
    """
    Call the fixed judge LLM with a system+user prompt.
    Reads API_BASE_URL and API_KEY fresh from env at call time
    so the judges' injected credentials are always used.
    """
    import openai

    api_key  = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
    api_base = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
    model    = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

    client = openai.OpenAI(base_url=api_base, api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=MAX_JUDGE_TOKENS,
            temperature=0.0,
            stream=False,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[JUDGE_ERROR: {e}]"


# ─────────────────────────────────────────────
# DETERMINISTIC VERIFIERS — Pure Python, no LLM needed
# Each verifier returns (score: float, passed: List[str], failed: List[str])
# ─────────────────────────────────────────────

def _verify_json_keys(judge_response: str, spec_value: dict) -> Tuple[float, List[str], List[str]]:
    """Check that the judge output is valid JSON with required keys and types."""
    passed, failed = [], []

    # Try to extract JSON from the response (handle ```json ... ``` blocks)
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", judge_response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r"\{.*\}", judge_response, re.DOTALL)
        json_str = json_match.group(0) if json_match else judge_response

    try:
        data = json.loads(json_str)
        passed.append("valid_json")
    except json.JSONDecodeError:
        failed.append("valid_json")
        failed.extend([f"has_{k}" for k in spec_value["required_keys"]])
        return 0.0, passed, failed

    for key in spec_value["required_keys"]:
        if key in data:
            passed.append(f"has_{key}")
        else:
            failed.append(f"has_{key}")

    for key, expected_type in spec_value.get("type_checks", {}).items():
        if key not in data:
            continue
        val = data[key]
        type_ok = False
        if expected_type == "int":
            type_ok = isinstance(val, int) and not isinstance(val, bool)
        elif expected_type == "float":
            type_ok = isinstance(val, (int, float)) and not isinstance(val, bool)
        elif expected_type == "bool":
            type_ok = isinstance(val, bool)
        elif expected_type == "str":
            type_ok = isinstance(val, str)

        check_name = f"{key}_correct_type"
        if type_ok:
            passed.append(check_name)
        else:
            failed.append(check_name)

    total_checks = len(spec_value["required_keys"]) + len(spec_value.get("type_checks", {})) + 1  # +1 for valid_json
    score = len(passed) / total_checks
    return round(score, 4), passed, failed


def _verify_code_tests(judge_response: str, spec_value: dict) -> Tuple[float, List[str], List[str]]:
    """Extract Python function from response and run unit tests."""
    passed, failed = [], []

    code_match = re.search(r"```(?:python)?\s*(.*?)```", judge_response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        def_match = re.search(r"(def\s+\w+.*?)(?=\n\n|\Z)", judge_response, re.DOTALL)
        code = def_match.group(1) if def_match else ""

    if not code:
        failed.append("code_extracted")
        failed.extend([f"test_{i+1}" for i in range(len(spec_value["test_cases"]))])
        return 0.0, passed, failed

    passed.append("code_extracted")

    fn = spec_value["function_name"]
    if f"def {fn}" in code:
        passed.append("correct_function_name")
    else:
        failed.append("correct_function_name")
        failed.extend([f"test_{i+1}" for i in range(len(spec_value["test_cases"]))])
        total = len(spec_value["test_cases"]) + 2
        return len(passed) / total, passed, failed

    namespace: Dict[str, Any] = {}
    try:
        exec(compile(code, "<judge_code>", "exec"), namespace)
    except Exception as e:
        failed.append(f"code_runs (error: {type(e).__name__})")
        failed.extend([f"test_{i+1}" for i in range(len(spec_value["test_cases"]))])
        total = len(spec_value["test_cases"]) + 2
        return len(passed) / total, passed, failed

    passed.append("code_runs")

    for i, (call_str, expected) in enumerate(spec_value["test_cases"]):
        test_name = f"test_{i+1}_{call_str[:30]}"
        try:
            result = eval(call_str, namespace)
            if result == expected:
                passed.append(test_name)
            else:
                failed.append(f"{test_name} (got {result!r}, expected {expected!r})")
        except Exception as e:
            failed.append(f"{test_name} (exception: {type(e).__name__})")

    total = len(spec_value["test_cases"]) + 3  # extracted + function_name + runs
    return round(len(passed) / total, 4), passed, failed


def _verify_phrase_match(judge_response: str, spec_value: dict) -> Tuple[float, List[str], List[str]]:
    """Check that required phrases appear in the judge's response."""
    passed, failed = [], []
    for phrase in spec_value["required_phrases"]:
        if phrase in judge_response:
            passed.append(f"has_phrase: '{phrase[:40]}'")
        else:
            failed.append(f"missing_phrase: '{phrase[:40]}'")
    score = len(passed) / len(spec_value["required_phrases"])
    return round(score, 4), passed, failed


def _verify_length_range(judge_response: str, spec_value: dict) -> Tuple[float, List[str], List[str]]:
    """Check response length and forbidden patterns."""
    passed, failed = [], []
    checks_total = 0

    length = len(judge_response)
    checks_total += 2

    if length >= spec_value["min_chars"]:
        passed.append(f"min_length ({length} >= {spec_value['min_chars']})")
    else:
        failed.append(f"min_length ({length} < {spec_value['min_chars']})")

    if length <= spec_value["max_chars"]:
        passed.append(f"max_length ({length} <= {spec_value['max_chars']})")
    else:
        failed.append(f"max_length ({length} > {spec_value['max_chars']})")

    for pattern in spec_value.get("forbidden_patterns", []):
        checks_total += 1
        if not re.search(pattern, judge_response):
            passed.append(f"no_forbidden: {pattern}")
        else:
            failed.append(f"has_forbidden: {pattern}")

    return round(len(passed) / checks_total, 4), passed, failed


def _verify_refusal(judge_response: str, spec_value: dict) -> Tuple[float, List[str], List[str]]:
    """Check that required refusal signals are present and forbidden content is absent."""
    passed, failed = [], []
    checks_total = 0

    required_any = spec_value.get("required_phrases_any", [])
    if required_any:
        checks_total += 1
        if any(p.lower() in judge_response.lower() for p in required_any):
            passed.append("has_required_signal")
        else:
            failed.append(f"missing_required_signal (needs one of: {required_any})")

    for pattern in spec_value.get("forbidden_patterns", []):
        checks_total += 1
        if not re.search(pattern, judge_response, re.IGNORECASE):
            passed.append(f"no_forbidden: {pattern}")
        else:
            failed.append(f"has_forbidden: {pattern}")

    return round(len(passed) / checks_total, 4) if checks_total else 0.0, passed, failed


VERIFIERS = {
    "json_keys": _verify_json_keys,
    "code_tests": _verify_code_tests,
    "phrase_match": _verify_phrase_match,
    "length_range": _verify_length_range,
    "refusal": _verify_refusal,
}


class PromptOptimizerEnvironment(Environment):
    """
    Prompt Optimization RL Environment.

    The agent iteratively rewrites a system prompt to make a fixed judge LLM
    produce a target structured output. Reward is fully deterministic Python verification.
    """

    def __init__(self):
        self._state: Optional[PromptOptimizerState] = None

    def reset(self, seed=None, episode_id=None, **kwargs) -> PromptOptimizerObservation:
        """Load a new task and return the initial observation."""
        if seed is not None:
            task = TASK_BANK[seed % len(TASK_BANK)]
        else:
            task = random.choice(TASK_BANK)

        initial_prompt = task["initial_bad_prompt"]

        self._state = PromptOptimizerState(
            episode_id=str(uuid.uuid4())[:8],
            step_count=0,
            task_id=task["task_id"],
            best_score_so_far=0.0,
            current_prompt=initial_prompt,
            task_description=task["description"],
            target_spec_type=task["spec_type"],
            target_spec_value=task["spec_value"],
            initial_bad_prompt=initial_prompt,
            prompt_history=[initial_prompt],
            reward_history=[],
        )

        # Score the initial bad prompt so agent gets useful first observation
        judge_response = _call_judge(
            initial_prompt,
            task["spec_value"].get("context_prompt_for_judge", "Do the task."),
        )
        score, passed, failed = VERIFIERS[task["spec_type"]](judge_response, task["spec_value"])
        self._state.reward_history.append(score)

        return PromptOptimizerObservation(
            task_description=task["description"],
            target_spec_type=task["spec_type"],
            target_spec_value=task["spec_value"],
            current_prompt=initial_prompt,
            judge_response=judge_response,
            checks_passed=passed,
            checks_failed=failed,
            step_reward=score,
            step_number=0,
            done=False,
            success=score >= 1.0,
            prompt_history=[initial_prompt],
            reward_history=[score],
        )

    def step(self, action: PromptOptimizerAction, timeout_s=None, **kwargs) -> PromptOptimizerObservation:
        """
        Agent provides a rewritten prompt.
        Environment tests it against the judge and returns deterministic score.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        self._state.step_count += 1
        new_prompt = action.rewritten_prompt.strip()
        self._state.current_prompt = new_prompt
        self._state.prompt_history.append(new_prompt)

        context_user_msg = self._state.target_spec_value.get(
            "context_prompt_for_judge", "Do the task."
        )
        judge_response = _call_judge(new_prompt, context_user_msg)

        spec_type = self._state.target_spec_type
        spec_value = self._state.target_spec_value
        score, passed, failed = VERIFIERS[spec_type](judge_response, spec_value)

        self._state.reward_history.append(score)
        self._state.best_score_so_far = max(self._state.best_score_so_far, score)

        success = score >= 1.0
        done = success or self._state.step_count >= MAX_STEPS

        return PromptOptimizerObservation(
            task_description=self._state.task_description,
            target_spec_type=spec_type,
            target_spec_value=spec_value,
            current_prompt=new_prompt,
            judge_response=judge_response,
            checks_passed=passed,
            checks_failed=failed,
            step_reward=score,
            step_number=self._state.step_count,
            done=done,
            success=success,
            prompt_history=self._state.prompt_history[-3:],
            reward_history=self._state.reward_history,
        )

    @property
    def state(self) -> PromptOptimizerState:  # type: ignore[override]
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state
