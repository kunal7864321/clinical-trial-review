from fastapi import FastAPI
from environment.env import ClinicalTrialEnv, Action

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

@app.get("/")
def root():
    return {"message": "Clinical Trial Review Environment is running"}