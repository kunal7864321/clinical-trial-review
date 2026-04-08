import os
import sys
import json
import re
import urllib.error
import urllib.parse
import urllib.request

try:
    import requests
except ModuleNotFoundError:
    requests = None

from openai import OpenAI


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HF_API_KEY")
    or os.environ.get("HuggingFaceHubApiToken")
)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BASE_URL = os.environ.get("ENV_URL", "http://127.0.0.1:7860")

if HF_TOKEN:
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
else:
    client = None
    print("[WARN] No HF_TOKEN found, LLM calls will use fallback", file=sys.stderr)


def log_start(task_id, trial_id):
    print(f"[START] task_id={task_id} trial_id={trial_id}")


def log_step(task_id, step_number, action, reward, done, feedback):
    print(
        "[STEP] "
        f"task_id={task_id} "
        f"step={step_number} "
        f"action={action.get('action_type')} "
        f"target={action.get('target_section')} "
        f"reward={reward:.3f} "
        f"done={str(done).lower()} "
        f"feedback={json.dumps(feedback) if feedback else 'null'}"
    )


def log_end(task_id, final_score):
    print(f"[END] task_id={task_id} final_score={final_score:.3f}")


def _build_url(url, params):
    if not params:
        return url
    query = urllib.parse.urlencode(params, doseq=True)
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{query}"


def _post_with_urllib(url, params=None, payload=None, timeout=30):
    request_url = _build_url(url, params)
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(request_url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            body = resp.read().decode(charset)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach {request_url}: {exc}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {request_url}") from exc


def post_env(path, params=None, payload=None, timeout=30):
    url = f"{BASE_URL}{path}"
    try:
        if requests is not None:
            response = requests.post(url, params=params, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        return _post_with_urllib(url, params=params, payload=payload, timeout=timeout)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Environment request failed for {url}: {exc}") from exc


def ask_agent(task_description, protocol_text, step_number):
    """Ask the LLM what action to take given the current observation."""
    if client is None:
        return json.dumps(
            {
                "action_type": "flag_issue",
                "target_section": "unknown",
                "issue_description": "No LLM client available",
                "severity": "low",
            }
        )

    prompt = f"""You are an expert clinical trial protocol reviewer.

TASK: {task_description}

PROTOCOL:
{json.dumps(protocol_text, indent=2)}

STEP: {step_number}

You must respond with ONLY a valid JSON object with exactly these fields:
{{
  "action_type": "flag_issue" or "approve_section" or "recommend_amendment",
  "target_section": "name of the section you are acting on",
  "issue_description": "detailed explanation of the issue you found or why you are approving",
  "severity": "low" or "medium" or "high" or "critical"
}}

Do not include any text outside the JSON object."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as exc:
        print(f"[WARN] LLM request failed: {exc}", file=sys.stderr)
        return json.dumps(
            {
                "action_type": "flag_issue",
                "target_section": "unknown",
                "issue_description": f"LLM call failed: {exc}",
                "severity": "low",
            }
        )


def parse_action(raw_text):
    """Parses JSON from LLM response, handling markdown blocks and extra text."""
    try:
        # Try to find JSON in markdown blocks
        json_match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1).strip())
        
        # Try to find anything that looks like a JSON object
        json_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1).strip())

        return json.loads(raw_text.strip())
    except Exception as e:
        print(f"[WARN] Failed to parse action JSON: {e}")
        return {
            "action_type": "flag_issue",
            "target_section": "unknown",
            "issue_description": f"Could not parse agent response: {raw_text[:100]}...",
            "severity": "low",
        }


def _clamp_score(value):
    """Clamp a score to be strictly within (0, 1) — never 0.0 or 1.0."""
    return float(max(0.001, min(0.999, value)))


def run_task(task_id):
    """Run one full episode for a given task and return the final score."""
    try:
        obs = post_env("/reset", params={"task_id": task_id})
    except Exception as exc:
        print(f"[WARN] Failed to reset environment: {exc}", file=sys.stderr)
        log_start(task_id, "unknown")
        log_end(task_id, _clamp_score(0.001))
        return _clamp_score(0.001)

    log_start(task_id, obs["trial_id"])

    done = False
    step = 0
    max_steps = 20
    env_total_reward = _clamp_score(0.001)

    while not done and step < max_steps:
        try:
            raw = ask_agent(
                obs["task_description"], obs["protocol_text"], obs["step_number"]
            )
            action = parse_action(raw)

            result_data = post_env("/step", payload=action)

            reward = result_data["reward"]["score"]
            feedback = result_data["reward"]["feedback"]
            # Use the environment's clamped total_reward (already in (0,1))
            env_total_reward = result_data["info"]["total_reward"]
            done = result_data["done"]
            obs = result_data["observation"]

            log_step(
                task_id,
                step_number=step + 1,
                action=action,
                reward=_clamp_score(reward),
                done=done,
                feedback=feedback,
            )
            step += 1
        except Exception as exc:
            print(f"[WARN] Step failed: {exc}", file=sys.stderr)
            done = True

    # Final clamp — guarantees score is always in (0, 1)
    final_score = _clamp_score(env_total_reward)
    log_end(task_id, final_score)
    return final_score


def main():
    scores = {}
    for task_id in [1, 2, 3]:
        scores[f"task_{task_id}"] = run_task(task_id)
    average = _clamp_score(sum(scores.values()) / len(scores))
    print("[SUMMARY] average_score={:.3f}".format(average))


if __name__ == "__main__":
    main()

