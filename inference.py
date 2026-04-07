import os
import json
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
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("HF_API_KEY"))
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BASE_URL = os.environ.get("ENV_URL", "http://127.0.0.1:7860")

if HF_TOKEN:
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
else:
    client = None


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
        raise RuntimeError(
            "OpenAI client not initialized. Set HF_TOKEN or HF_API_KEY environment variable."
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
    except Exception as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc
    return response.choices[0].message.content


def parse_action(raw_text):
    """Parse the LLM response into a valid action dict."""
    try:
        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        return json.loads(raw_text.strip())
    except Exception:
        return {
            "action_type": "flag_issue",
            "target_section": "unknown",
            "issue_description": "Could not parse agent response",
            "severity": "low",
        }


def run_task(task_id):
    """Run one full episode for a given task and return the final score."""
    obs = post_env("/reset", params={"task_id": task_id})

    log_start(task_id, obs["trial_id"])

    total_reward = 0.0
    done = False
    step = 0
    max_steps = 10  # limit steps in inference to save time

    while not done and step < max_steps:
        # ask the agent what to do
        raw = ask_agent(
            obs["task_description"], obs["protocol_text"], obs["step_number"]
        )
        action = parse_action(raw)

        # send action to environment
        result_data = post_env("/step", payload=action)

        reward = result_data["reward"]["score"]
        feedback = result_data["reward"]["feedback"]
        total_reward += reward
        done = result_data["done"]
        obs = result_data["observation"]

        log_step(
            task_id,
            step_number=step + 1,
            action=action,
            reward=reward,
            done=done,
            feedback=feedback,
        )
        step += 1

    final_score = max(0.001, min(0.999, total_reward))
    log_end(task_id, final_score)
    return final_score


def main():
    scores = {}
    for task_id in [1, 2, 3]:
        scores[f"task_{task_id}"] = run_task(task_id)
    average = sum(scores.values()) / len(scores)
    print("[SUMMARY] average_score={:.3f}".format(average))


if __name__ == "__main__":
    main()
