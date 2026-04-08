---
title: Prompt Optimizer RL Environment
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Prompt Optimizer RL Environment

An RL environment where an agent iteratively rewrites system prompts to make a fixed judge LLM produce specific structured outputs. Reward is fully deterministic Python verification.

## Tasks
- JSON structure (2 tasks)
- Code correctness (2 tasks)
- Phrase matching (2 tasks)
- Length constraints (1 task)
- Refusal behavior (1 task)

## Usage
```bash
POST /reset  # Start a new episode
POST /step   # Submit a rewritten prompt
GET  /health # Check server status
```

## Environment Variables
- `HF_TOKEN` — Hugging Face token
- `JUDGE_API_KEY` — API key for judge LLM
- `JUDGE_MODEL` — Judge model name (default: llama-3.1-8b-instant)
- `JUDGE_API_BASE` — Judge API base URL
