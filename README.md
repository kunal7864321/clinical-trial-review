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

An OpenEnv-compatible evaluation environment where agents review full-length clinical trial protocols, highlight regulatory issues, and explain their reasoning. The repo contains the simulator, FastAPI surface used inside the Hugging Face Space, and a reference inference loop for benchmarking models.

## Highlights
- Multi-task environment covering missing sections, unsafe dosages, and internal contradictions – each with tailored rewards
- FastAPI server (`app.py`) exposes `/reset`, `/step`, and `/state` endpoints that plug directly into the Space UI or custom agents
- Ground-truth rich synthetic protocols in `environment/data/protocols/` with per-task annotations used for scoring
- Baseline evaluator (`inference.py`) that can call any OpenAI-compatible endpoint (Hugging Face Inference, OpenAI, local gateways, etc.)

## Repository Layout
| Path | Purpose |
|------|---------|
| `environment/env.py` | Core `ClinicalTrialEnv` class, schemas, reward functions |
| `environment/data/protocols/` | JSON protocols with sections, dosage info, and contradictions |
| `app.py` | FastAPI wiring for `/reset`, `/step`, `/state` |
| `server/app.py` | Uvicorn entry point used by Docker/HF Space |
| `inference.py` | Baseline agent loop + logging utilities |
| `Dockerfile` | Minimal image that runs `uvicorn app:app` on port 7860 |

## Quickstart
1. **Install dependencies**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Run the API locally**
   ```bash
   uvicorn server.app:main --host 0.0.0.0 --port 7860
   ```
   The Hugging Face Space uses the exact same command through the bundled Dockerfile.
3. **(Optional) Point clients at the API** by setting `ENV_URL=http://127.0.0.1:7860` (default) before running agents/tests.

## API Quick Reference
| Method | Path | Body | Response |
|--------|------|------|----------|
| `POST` | `/reset?task_id=1` | none | First observation for the requested task |
| `POST` | `/step` | `{action_type, target_section, issue_description, severity}` | Next observation + `reward`, `done`, `info.total_reward` |
| `GET` | `/state` | none | Internal simulator state (for debugging/visualization) |

Example:
```bash
curl -X POST "http://localhost:7860/reset?task_id=2"
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{
           "action_type": "flag_issue",
           "target_section": "dosage",
           "issue_description": "DrugY dosage is 2400mg/day, above the 2000mg/day limit.",
           "severity": "high"
         }'
```

## Tasks & Dataset
| Task ID | Focus | Success Criteria |
|---------|-------|------------------|
| 1 | Missing Section Detection | Flag every required protocol section that is absent, optionally approve sections that are present |
| 2 | Dosage Safety Compliance | Identify drug doses that exceed `MAX_DRUG_DOSES` in `environment/data/rules.py` and justify severity |
| 3 | Contradiction Detection | Point out conflicting statements between sections with references to both sides |

Protocols are sampled at reset time and contain `sections` plus a `ground_truth` blob (missing sections, unsafe dosages, contradictions) that drives the reward.

## Observation, Action, Reward Shapes
- **Observation** (`Observation` model): `trial_id`, `protocol_text`, `task_description`, `step_number`, `available_actions`
- **Action** (`Action` model): JSON object with `action_type` (`flag_issue`, `approve_section`, `recommend_amendment`), `target_section`, `issue_description`, `severity`
- **Reward** (`Reward` model): `score` in `[0, 1]`, `breakdown` dict listing reward components, textual `feedback`

Reward weights differ per task:
- Task 1: +0.3 for correct missing-section flag, +0.1 bonus for detailed explanation, -0.1 for false positives
- Task 2: +0.4 for correct unsafe dosage, +0.2 for detailed rationale, +0.1 for correct high severity, up to -0.15 when wrong
- Task 3: +0.4 when both contradictory sections are referenced, +0.3 for long-form explanation, -0.1 for spurious flags

## Baseline Agent & Benchmarking
The reference agent in `inference.py` loops over the three tasks, logs every step, and prints a summary average. It talks to the environment via HTTP and to an OpenAI-compatible text model for decisions.

1. Start the API (local or inside the Space) and ensure it is reachable on `ENV_URL`.
2. Export credentials and overrides:
   ```bash
   export HF_TOKEN="hf_xxx"                        # required for Hugging Face text models
   export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"   # or any compatible chat-completions model
   export API_BASE_URL="https://router.huggingface.co/v1"  # defaults to OpenAI API
   export ENV_URL="https://clinical-trial-review.hf.space" # optional when running against the Space
   ```
3. Run `python inference.py` to stream step-level logs and an overall average score. Logs follow the `START/STEP/END` pattern expected by OpenEnv submissions.

## Container & Hugging Face Space
- The Space uses `sdk: docker` with the provided `Dockerfile`. Any local change will be replicated by rebuilding and pushing the repo.
- You can validate the container locally with:
  ```bash
  docker build -t clinical-trial-review .
  docker run -p 7860:7860 clinical-trial-review
  ```
  Then run agents against `http://localhost:7860`.

### Deploying to a new Space
1. **Authenticate**
   ```bash
   pip install -U "huggingface_hub[cli]"
   huggingface-cli login
   ```
2. **Create the Space** (replace `ORG/clinical-trial-review` with your handle):
   ```bash
   huggingface-cli repo create ORG/clinical-trial-review \
       --type space --space-sdk docker
   ```
   You can also create it from https://huggingface.co/spaces/new by choosing **Docker** as the SDK.
3. **Push this repo**
   ```bash
   git remote add hf https://huggingface.co/spaces/ORG/clinical-trial-review
   git push hf main
   ```
   Every subsequent push to `hf` will trigger a rebuild using the bundled `Dockerfile`.
4. **Configure secrets (optional but recommended)** in **Settings → Variables & secrets** if you want the bundled `inference.py` to call hosted models:
   - `HF_TOKEN` – token with access to the selected text model
   - `MODEL_NAME`, `API_BASE_URL`, `ENV_URL` – override defaults consumed by `inference.py`
5. **Monitor the build** from the Space **Logs** tab. Once the container reports `Application running on 0.0.0.0:7860`, the environment is live at `https://ORG-clinical-trial-review.hf.space`.

## Additional Notes
- `openenv.yaml` documents the task metadata consumed by the OpenEnv leaderboard.
- `pyproject.toml` exposes a `server` entry point (`python -m server.app`) if you prefer `pip install .` workflows.
- For troubleshooting, call `GET /state` to inspect accumulated actions and rewards.
