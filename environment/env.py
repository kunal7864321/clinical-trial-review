import json
import os
import random
from typing import Any
from pydantic import BaseModel
from environment.data.rules import REQUIRED_SECTIONS, MAX_DRUG_DOSES, AGE_GUIDELINES


# ─── TYPED MODELS (required by OpenEnv spec) ───

class Observation(BaseModel):
    trial_id: str
    protocol_text: dict
    task_description: str
    step_number: int
    available_actions: list[str]

class Action(BaseModel):
    action_type: str        # "flag_issue", "approve_section", "recommend_amendment"
    target_section: str     # which section this action is about
    issue_description: str  # agent's explanation
    severity: str           # "low", "medium", "high", "critical"

class Reward(BaseModel):
    score: float
    breakdown: dict
    feedback: str


# ─── MAIN ENVIRONMENT CLASS ───

class ClinicalTrialEnv:
    def __init__(self):
        self.current_protocol = None
        self.current_task_id = None
        self.step_count = 0
        self.agent_actions = []
        self.max_steps = 20
        self.total_reward = 0.0
        self.protocols = self._load_protocols()

    def _load_protocols(self):
        protocols = []
        protocols_dir = os.path.join(
            os.path.dirname(__file__), "data", "protocols"
        )
        for filename in os.listdir(protocols_dir):
            if filename.endswith(".json"):
                with open(os.path.join(protocols_dir, filename)) as f:
                    protocols.append(json.load(f))
        return protocols

    def reset(self, task_id: int = 1) -> Observation:
        self.current_protocol = random.choice(self.protocols)
        self.current_task_id = task_id
        self.step_count = 0
        self.agent_actions = []
        self.total_reward = 0.0

        task_descriptions = {
            1: "TASK 1 - EASY: Read this clinical trial protocol carefully. Identify all MISSING required sections. Required sections are: objectives, inclusion_criteria, exclusion_criteria, dosage, adverse_event_reporting, statistical_analysis_plan, withdrawal_criteria, informed_consent. Flag each missing section using action_type='flag_issue'.",
            2: "TASK 2 - MEDIUM: Read this clinical trial protocol carefully. Identify any drug dosages that exceed the maximum allowed limits. DrugX max=500mg/day, DrugY max=2000mg/day, DrugZ max=750mg/day, DrugA max=1000mg/day, DrugB max=300mg/day. Flag each unsafe dosage using action_type='flag_issue'.",
            3: "TASK 3 - HARD: Read this clinical trial protocol carefully. Identify any internal contradictions between different sections of the protocol. A contradiction is when two sections say things that conflict with each other. Flag each contradiction using action_type='flag_issue' and explain both sections that conflict.",
        }

        return Observation(
            trial_id=self.current_protocol["trial_id"],
            protocol_text=self.current_protocol["sections"],
            task_description=task_descriptions[task_id],
            step_number=0,
            available_actions=["flag_issue", "approve_section", "recommend_amendment"]
        )

    def step(self, action: Action):
        self.step_count += 1
        self.agent_actions.append(action.model_dump())

        reward_score, breakdown, feedback = self._calculate_reward(action)
        self.total_reward += reward_score

        done = self.step_count >= self.max_steps

        next_obs = Observation(
            trial_id=self.current_protocol["trial_id"],
            protocol_text=self.current_protocol["sections"],
            task_description=f"Continue reviewing. Step {self.step_count} of {self.max_steps}.",
            step_number=self.step_count,
            available_actions=["flag_issue", "approve_section", "recommend_amendment"]
        )

        reward = Reward(
            score=round(reward_score, 3),
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
            "actions_taken": len(self.agent_actions),
            "agent_actions": self.agent_actions
        }

    def _calculate_reward(self, action: Action):
        breakdown = {}
        feedback_parts = []
        score = 0.0

        if self.current_task_id == 1:
            score, breakdown, feedback_parts = self._reward_task1(action)
        elif self.current_task_id == 2:
            score, breakdown, feedback_parts = self._reward_task2(action)
        elif self.current_task_id == 3:
            score, breakdown, feedback_parts = self._reward_task3(action)

        final_score = max(0.0, min(1.0, score))
        return final_score, breakdown, " | ".join(feedback_parts)

    def _reward_task1(self, action: Action):
        score = 0.0
        breakdown = {}
        feedback = []

        missing = self.current_protocol["ground_truth"]["missing_sections"]

        if action.action_type == "flag_issue":
            if action.target_section in missing:
                score += 0.3
                breakdown["correct_flag"] = 0.3
                feedback.append(f"Correct: {action.target_section} is indeed missing")
                if len(action.issue_description) > 50:
                    score += 0.1
                    breakdown["explanation_bonus"] = 0.1
                    feedback.append("Good explanation provided")
            else:
                score -= 0.1
                breakdown["false_positive"] = -0.1
                feedback.append(f"Incorrect: {action.target_section} is not missing")
        elif action.action_type == "approve_section":
            sections = self.current_protocol["sections"]
            if action.target_section in sections:
                score += 0.1
                breakdown["correct_approval"] = 0.1
                feedback.append(f"Correct: {action.target_section} exists and approved")
            else:
                score -= 0.1
                breakdown["wrong_approval"] = -0.1
                feedback.append("Approved a section that does not exist")

        return score, breakdown, feedback

    def _reward_task2(self, action: Action):
        score = 0.0
        breakdown = {}
        feedback = []

        unsafe = self.current_protocol["ground_truth"]["unsafe_dosages"]
        unsafe_drugs = [u["drug"] for u in unsafe]

        if action.action_type == "flag_issue":
            flagged_drug = None
            for drug in MAX_DRUG_DOSES:
                if drug.lower() in action.target_section.lower() or drug.lower() in action.issue_description.lower():
                    flagged_drug = drug
                    break

            if flagged_drug and flagged_drug in unsafe_drugs:
                score += 0.4
                breakdown["correct_unsafe_flag"] = 0.4
                feedback.append(f"Correct: {flagged_drug} dosage is unsafe")
                if len(action.issue_description) > 50:
                    score += 0.2
                    breakdown["explanation_bonus"] = 0.2
                    feedback.append("Good explanation with details")
                if action.severity in ["high", "critical"]:
                    score += 0.1
                    breakdown["severity_bonus"] = 0.1
                    feedback.append("Correctly marked as high severity")
            elif flagged_drug and flagged_drug not in unsafe_drugs:
                score -= 0.15
                breakdown["false_positive"] = -0.15
                feedback.append(f"Incorrect: {flagged_drug} dosage is within safe limits")
            else:
                score -= 0.1
                breakdown["unclear_flag"] = -0.1
                feedback.append("Flag did not clearly identify which drug is unsafe")

        return score, breakdown, feedback

    def _reward_task3(self, action: Action):
        score = 0.0
        breakdown = {}
        feedback = []

        contradictions = self.current_protocol["ground_truth"]["contradictions"]

        if action.action_type == "flag_issue":
            matched = False
            for contradiction in contradictions:
                sec_a = contradiction["section_a"].lower()
                sec_b = contradiction["section_b"].lower()
                desc_words = contradiction["description"].lower().split()
                key_words = [w for w in desc_words if len(w) > 5][:5]

                desc_lower = action.issue_description.lower()
                keyword_matches = sum(1 for w in key_words if w in desc_lower)

                sections_mentioned = (sec_a in desc_lower or sec_a in action.target_section.lower()) and \
                                     (sec_b in desc_lower or sec_b in action.target_section.lower())

                if keyword_matches >= 2 or sections_mentioned:
                    matched = True
                    score += 0.4
                    breakdown["contradiction_found"] = 0.4
                    feedback.append(f"Correct contradiction identified between {sec_a} and {sec_b}")
                    if len(action.issue_description) > 80:
                        score += 0.3
                        breakdown["detailed_explanation"] = 0.3
                        feedback.append("Excellent detailed explanation")
                    break

            if not matched:
                score -= 0.1
                breakdown["false_positive"] = -0.1
                feedback.append("Contradiction flagged does not match known issues in this protocol")

        return score, breakdown, feedback