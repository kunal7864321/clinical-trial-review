import os
import json
import requests
from openai import OpenAI


API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
BASE_URL = os.environ.get("ENV_URL", "http://127.0.0.1:8000")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


def ask_agent(task_description, protocol_text, step_number):
    """Ask the LLM what action to take given the current observation."""
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

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.1
    )
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
            "severity": "low"
        }


def run_task(task_id):
    """Run one full episode for a given task and return the final score."""
    print(f"\n{'='*50}")
    print(f"Running Task {task_id}...")
    print(f"{'='*50}")

    # reset the environment
    response = requests.post(f"{BASE_URL}/reset", params={"task_id": task_id})
    obs = response.json()

    print(f"Trial: {obs['trial_id']}")
    print(f"Task: {obs['task_description'][:80]}...")

    total_reward = 0.0
    done = False
    step = 0
    max_steps = 10  # limit steps in inference to save time

    while not done and step < max_steps:
        # ask the agent what to do
        raw = ask_agent(
            obs["task_description"],
            obs["protocol_text"],
            obs["step_number"]
        )
        action = parse_action(raw)

        print(f"\nStep {step + 1}: {action['action_type']} → {action['target_section']}")

        # send action to environment
        result = requests.post(f"{BASE_URL}/step", json=action)
        result_data = result.json()

        reward = result_data["reward"]["score"]
        feedback = result_data["reward"]["feedback"]
        total_reward += reward
        done = result_data["done"]
        obs = result_data["observation"]

        print(f"Reward: {reward} | Feedback: {feedback}")
        step += 1

    final_score = max(0.0, min(1.0, total_reward))
    print(f"\nTask {task_id} Final Score: {final_score:.3f}")
    return final_score


def main():
    print("Clinical Trial Review Environment — Baseline Inference")
    print("Model:", MODEL_NAME)
    print("API:", API_BASE_URL)

    scores = {}
    for task_id in [1, 2, 3]:
        scores[f"task_{task_id}"] = run_task(task_id)

    print(f"\n{'='*50}")
    print("FINAL BASELINE SCORES")
    print(f"{'='*50}")
    for task, score in scores.items():
        print(f"{task}: {score:.3f}")
    print(f"Average: {sum(scores.values()) / len(scores):.3f}")


if __name__ == "__main__":
    main()