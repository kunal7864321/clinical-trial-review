---
title: Clinical Trial Review Environment
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - rl
  - agents
---

# Clinical Trial Review Environment

An environment for training and evaluating AI agents on clinical trial protocol review tasks.

## What It Does
Agents review simulated pharmaceutical clinical trial protocols and must identify safety violations, regulatory compliance failures, and internal contradictions — tasks that trained human reviewers spend days doing manually.

## Tasks
| Task | Difficulty | Description |
|------|-----------|-------------|
| Task 1 | Easy | Identify missing required sections |
| Task 2 | Medium | Flag drug dosages exceeding WHO/FDA limits |
| Task 3 | Hard | Detect internal contradictions between sections |

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| /reset | POST | Start new episode, returns observation |
| /step | POST | Submit action, returns reward + feedback |
| /state | GET | Returns current environment state |

## Observation Space
- `trial_id` — protocol identifier
- `protocol_text` — full protocol sections
- `task_description` — what the agent must do
- `step_number` — current step
- `available_actions` — list of valid action types

## Action Space
- `action_type` — flag_issue / approve_section / recommend_amendment
- `target_section` — which section the action applies to
- `issue_description` — agent's explanation
- `severity` — low / medium / high / critical

## Reward Function
- +0.1 to +0.4 for correct flags depending on task
- +0.1 to +0.3 bonus for detailed explanations
- -0.1 to -0.15 penalty for false positives

## Baseline Scores (Qwen2.5-72B-Instruct)
| Task | Score |
|------|-------|
| Task 1 — Missing Sections | 1.000 |
| Task 2 — Dosage Safety | 0.700 |
| Task 3 — Contradictions | 0.700 |
| **Average** | **0.800** |

## Setup
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Run Baseline Inference
```bash
export HF_TOKEN="your_token"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export API_BASE_URL="https://router.huggingface.co/v1"
python3 inference.py
```