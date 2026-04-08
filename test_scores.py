import json, requests, sys
sys.path.insert(0, ".")

BASE = "http://127.0.0.1:7860"
all_ok = True

print("=" * 60)
print("SCORE RANGE VALIDATION")
print("=" * 60)

for task_id in [1, 2, 3]:
    print(f"\n--- Task {task_id} ---")
    r = requests.post(f"{BASE}/reset", params={"task_id": task_id})
    obs = r.json()

    state = requests.get(f"{BASE}/state").json()
    tr = state["total_reward"]
    ok = 0 < tr < 1
    if not ok: all_ok = False
    print(f"  After reset: total_reward={tr} valid={ok}")

    actions = [
        {"action_type": "flag_issue", "target_section": "statistical_analysis_plan",
         "issue_description": "This required section is completely missing from the protocol document and must be added",
         "severity": "critical"},
        {"action_type": "flag_issue", "target_section": "withdrawal_criteria",
         "issue_description": "Missing withdrawal criteria section in protocol", "severity": "high"},
        {"action_type": "flag_issue", "target_section": "informed_consent",
         "issue_description": "Missing informed consent section needs to be added for compliance", "severity": "critical"},
        {"action_type": "flag_issue", "target_section": "DrugX",
         "issue_description": "DrugX dosage of 600mg exceeds the maximum allowed dose of 500mg per day", "severity": "critical"},
        {"action_type": "approve_section", "target_section": "objectives",
         "issue_description": "Section looks complete", "severity": "low"},
        {"action_type": "flag_issue", "target_section": "fake",
         "issue_description": "wrong", "severity": "low"},
        {"action_type": "recommend_amendment", "target_section": "dosage",
         "issue_description": "Needs amendment", "severity": "medium"},
    ]

    for i in range(20):
        action = actions[i % len(actions)]
        r = requests.post(f"{BASE}/step", json=action)
        data = r.json()
        rs = data["reward"]["score"]
        itr = data["info"]["total_reward"]
        ok_r = 0 < rs < 1
        ok_t = 0 < itr < 1
        if not ok_r:
            all_ok = False
            print(f"  FAIL Step {i+1}: reward.score={rs}")
        if not ok_t:
            all_ok = False
            print(f"  FAIL Step {i+1}: info.total_reward={itr}")
        if i == 0 or i == 19:
            print(f"  Step {i+1}: reward={rs:.6f} total={itr:.6f} valid={ok_r and ok_t}")

    state = requests.get(f"{BASE}/state").json()
    tr = state["total_reward"]
    ok = 0 < tr < 1
    if not ok: all_ok = False
    print(f"  Final state: total_reward={tr:.6f} valid={ok}")

print("\n--- Grader Direct Tests ---")
from environment.graders import grade_task1, grade_task2, grade_task3

tests = [
    ("t1_empty", grade_task1([], {"missing_sections": ["a", "b"]})),
    ("t1_perfect", grade_task1([{"action_type": "flag_issue", "target_section": "a"},
                                {"action_type": "flag_issue", "target_section": "b"}],
                               {"missing_sections": ["a", "b"]})),
    ("t1_no_missing", grade_task1([], {"missing_sections": []})),
    ("t1_50fp", grade_task1([{"action_type": "flag_issue", "target_section": "x"}] * 50,
                            {"missing_sections": ["a"]})),
    ("t2_empty", grade_task2([], {"unsafe_dosages": [{"drug": "DrugX"}]})),
    ("t2_perfect", grade_task2([{"action_type": "flag_issue", "target_section": "DrugX",
                                 "issue_description": "bad"}],
                               {"unsafe_dosages": [{"drug": "DrugX"}]})),
    ("t2_no_unsafe", grade_task2([], {"unsafe_dosages": []})),
    ("t3_empty", grade_task3([], {"contradictions": [{"section_a": "a", "section_b": "b",
                                                      "description": "test words here"}]})),
    ("t3_no_contra", grade_task3([], {"contradictions": []})),
]

for name, score in tests:
    ok = 0 < score < 1
    if not ok: all_ok = False
    print(f"  {name}: {score:.6f} valid={ok}")

print("\n" + "=" * 60)
print("RESULT:", "ALL PASS" if all_ok else "FAILURES DETECTED")
print("=" * 60)
