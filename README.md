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
- `API_KEY` — API key injected by the hackathon validator
- `API_BASE_URL` — LLM proxy base URL injected by the hackathon validator
- `MODEL_NAME` — Model name to call through the proxy
