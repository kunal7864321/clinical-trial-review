from fastapi import FastAPI
from environment.env import ClinicalTrialEnv, Action, clamp_score
from environment.graders import grade_task1, grade_task2, grade_task3

app = FastAPI(title="Clinical Trial Review Environment")
env = ClinicalTrialEnv()


@app.post("/reset")
def reset(task_id: int = 1):
    obs = env.reset(task_id)
    return obs


@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/state")
def state():
    return env.state()


@app.post("/grade")
def grade():
    """Grade the current episode using the grader functions."""
    if env.current_protocol is None:
        return {"task_scores": {1: 0.5, 2: 0.5, 3: 0.5}}

    gt = env.current_protocol["ground_truth"]
    actions = env.agent_actions

    graders = {1: grade_task1, 2: grade_task2, 3: grade_task3}

    if env.current_task_id and env.current_task_id in graders:
        score = graders[env.current_task_id](actions, gt)
        score = clamp_score(score)
        return {"task_id": env.current_task_id, "score": score}

    scores = {}
    for tid, grader_fn in graders.items():
        s = grader_fn(actions, gt)
        scores[tid] = clamp_score(s)
    return {"task_scores": scores}


@app.get("/")
def root():
    return {"message": "Clinical Trial Review Environment is running"}