import json
import os
import random
from typing import Any, Literal, List, Dict
from pydantic import BaseModel, Field, field_validator
from environment.data.rules import REQUIRED_SECTIONS, MAX_DRUG_DOSES, AGE_GUIDELINES

# ─── Score clamping helper ───
_SCORE_MIN = 0.000001
_SCORE_MAX = 0.99

def clamp_score(value: float) -> float:
    """Clamp a score to be strictly within (0, 1) — never 0.0 or 1.0."""
    return float(max(_SCORE_MIN, min(_SCORE_MAX, value)))

# ─── TYPED MODELS (required by OpenEnv spec) ───

class Observation(BaseModel):
    trial_id: str
    protocol_text: Dict[str, Any]
    task_description: str
    step_number: int
    available_actions: List[str]

class Action(BaseModel):
    action_type: Literal["flag_issue", "approve_section", "recommend_amendment"]
    target_section: str
    issue_description: str
    severity: Literal["low", "medium", "high", "critical"]

class Reward(BaseModel):
    score: float
    breakdown: Dict[str, float]
    feedback: str

    @field_validator("score")
    @classmethod
    def enforce_score_range(cls, v: float) -> float:
        return clamp_score(v)

# ─── MAIN ENVIRONMENT CLASS ───

class ClinicalTrialEnv:
    def __init__(self):
        self.current_protocol = None
        self.current_task_id = None
        self.step_count = 0
        self.agent_actions = []
        self.max_steps = 20
        self.total_reward = clamp_score(0.000001)
        self.current_task_description = ""
        self.protocols = self._load_protocols()

    def _load_protocols(self):
        protocols = []
        base_dir = os.path.dirname(__file__)
        protocols_dir = os.path.join(base_dir, "data", "protocols")
        
        if not os.path.exists(protocols_dir):
            return []

        for filename in os.listdir(protocols_dir):
            if filename.endswith(".json"):
                with open(os.path.join(protocols_dir, filename)) as f:
                    protocols.append(json.load(f))
        return protocols

    def reset(self, task_id: int = 1) -> Observation:
        if not self.protocols:
            raise ValueError("No protocols found in data/protocols")
            
        self.current_protocol = random.choice(self.protocols)
        self.current_task_id = task_id
        self.total_reward = clamp_score(0.000001)
        self.step_count = 0
        self.agent_actions = []

        task_descriptions = {
            1: "TASK 1 - EASY: Identify all MISSING required sections in the clinical trial protocol. Required sections are: objectives, inclusion_criteria, exclusion_criteria, dosage, adverse_event_reporting, statistical_analysis_plan, withdrawal_criteria, informed_consent. Flag each missing section using action_type='flag_issue'.",
            2: "TASK 2 - MEDIUM: Identify any drug dosages that exceed the maximum allowed limits (DrugX: 500mg, DrugY: 2000mg, DrugZ: 750mg, DrugA: 1000mg, DrugB: 300mg). Flag each unsafe dosage using action_type='flag_issue'.",
            3: "TASK 3 - HARD: Identify internal contradictions between different sections of the protocol. Flag each contradiction using action_type='flag_issue' and explain the conflict.",
        }

        self.current_task_description = task_descriptions.get(task_id, "Analyze the protocol for issues.")
        
        return Observation(
            trial_id=self.current_protocol["trial_id"],
            protocol_text=self.current_protocol["sections"],
            task_description=self.current_task_description,
            step_number=0,
            available_actions=["flag_issue", "approve_section", "recommend_amendment"],
        )

    def step(self, action: Action):
        self.step_count += 1
        self.agent_actions.append(action.model_dump())

        reward_score, breakdown, feedback = self._calculate_reward(action)
        self.total_reward = clamp_score(self.total_reward + reward_score)

        done = self.step_count >= self.max_steps

        next_obs = Observation(
            trial_id=self.current_protocol["trial_id"],
            protocol_text=self.current_protocol["sections"],
            task_description=self.current_task_description,
            step_number=self.step_count,
            available_actions=["flag_issue", "approve_section", "recommend_amendment"],
        )

        reward = Reward(
            score=reward_score,
            breakdown=breakdown,
            feedback=feedback
        )

        return next_obs, reward, done, {"total_reward": self.total_reward}

    def state(self):
        return {
            "trial_id": self.current_protocol["trial_id"] if self.current_protocol else None,
            "current_task_id": self.current_task_id,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "total_reward": self.total_reward,
            "agent_actions": self.agent_actions,
        }

    def _calculate_reward(self, action: Action):
        if self.current_task_id == 1:
            return self._reward_task1(action)
        elif self.current_task_id == 2:
            return self._reward_task2(action)
        elif self.current_task_id == 3:
            return self._reward_task3(action)
        return 0.000001, {}, "Unknown task"

    def _reward_task1(self, action: Action):
        missing = self.current_protocol["ground_truth"].get("missing_sections", [])
        if action.action_type == "flag_issue" and action.target_section in missing:
            return 0.3, {"correct_flag": 0.3}, f"Correct: {action.target_section} is missing"
        elif action.action_type == "flag_issue":
            return -0.05, {"false_positive": -0.05}, f"Incorrect: {action.target_section} is present"
        return 0.000001, {}, "No reward for this action"

    def _reward_task2(self, action: Action):
        unsafe = self.current_protocol["ground_truth"].get("unsafe_dosages", [])
        unsafe_drugs = [u["drug"].lower() for u in unsafe]
        
        if action.action_type == "flag_issue":
            found_drug = None
            for drug in MAX_DRUG_DOSES:
                if drug.lower() in action.target_section.lower() or drug.lower() in action.issue_description.lower():
                    found_drug = drug.lower()
                    break
            
            if found_drug and found_drug in unsafe_drugs:
                return 0.4, {"correct_dosage_flag": 0.4}, f"Correct: {found_drug} dosage is unsafe"
            elif found_drug:
                return -0.05, {"false_positive": -0.05}, f"Incorrect: {found_drug} dosage is safe"
        return 0.000001, {}, "No reward for this action"

    def _reward_task3(self, action: Action):
        contradictions = self.current_protocol["ground_truth"].get("contradictions", [])
        if action.action_type == "flag_issue":
            for c in contradictions:
                sec_a, sec_b = c["section_a"].lower(), c["section_b"].lower()
                desc = action.issue_description.lower()
                if (sec_a in desc or sec_a in action.target_section.lower()) and \
                   (sec_b in desc or sec_b in action.target_section.lower()):
                    return 0.5, {"contradiction_found": 0.5}, "Correct contradiction identified"
            return -0.05, {"false_positive": -0.05}, "No matching contradiction found"
        return 0.000001, {}, "No reward for this action"
