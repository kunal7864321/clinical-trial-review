"""Graders for clinical trial review tasks.

Every grader returns a score strictly in (0, 1) — never 0.0, never 1.0.
"""

# Strict bounds — scores must be > 0 and < 1
_SCORE_MIN = 0.01
_SCORE_MAX = 0.99


def _clamp(value: float) -> float:
    """Clamp a score to be strictly within (0, 1)."""
    clamped = max(_SCORE_MIN, min(_SCORE_MAX, value))
    return round(float(clamped), 6)


def grade_task1(agent_actions, ground_truth):
    """Missing section detection — score strictly in (0, 1)"""
    actual_missing = set(ground_truth["missing_sections"])

    if not actual_missing:
        return _clamp(0.95)

    correctly_flagged = set()
    false_positives = 0

    for action in agent_actions:
        if action["action_type"] == "flag_issue":
            section = action["target_section"]
            if section in actual_missing:
                correctly_flagged.add(section)
            else:
                false_positives += 1

    if len(actual_missing) == 0:
        return _clamp(0.5)

    precision_score = len(correctly_flagged) / len(actual_missing)
    penalty = false_positives * 0.05
    final = precision_score - penalty

    return _clamp(final)


def grade_task2(agent_actions, ground_truth):
    """Dosage safety compliance — score strictly in (0, 1)"""
    unsafe_dosages = ground_truth["unsafe_dosages"]

    if not unsafe_dosages:
        return _clamp(0.95)

    unsafe_drugs = [u["drug"] for u in unsafe_dosages]
    correctly_flagged = set()
    false_positives = 0

    for action in agent_actions:
        if action["action_type"] == "flag_issue":
            desc = action["issue_description"].lower()
            target = action["target_section"].lower()
            matched = False
            for drug in unsafe_drugs:
                if drug.lower() in desc or drug.lower() in target:
                    correctly_flagged.add(drug)
                    matched = True
                    break
            if not matched:
                false_positives += 1

    if len(unsafe_drugs) == 0:
        return _clamp(0.5)

    recall = len(correctly_flagged) / len(unsafe_drugs)
    penalty = false_positives * 0.05
    final = recall - penalty

    return _clamp(final)


def grade_task3(agent_actions, ground_truth):
    """Contradiction detection — score strictly in (0, 1)"""
    contradictions = ground_truth["contradictions"]

    if not contradictions:
        return _clamp(0.95)

    found_contradictions = set()
    false_positives = 0

    for action in agent_actions:
        if action["action_type"] == "flag_issue":
            desc = action["issue_description"].lower()
            target = action["target_section"].lower()
            matched = False

            for i, contradiction in enumerate(contradictions):
                sec_a = contradiction["section_a"].lower()
                sec_b = contradiction["section_b"].lower()
                desc_words = contradiction["description"].lower().split()
                key_words = [w for w in desc_words if len(w) > 5][:5]
                keyword_matches = sum(1 for w in key_words if w in desc)
                sections_mentioned = (sec_a in desc or sec_a in target) and (
                    sec_b in desc or sec_b in target
                )

                if keyword_matches >= 2 or sections_mentioned:
                    found_contradictions.add(i)
                    matched = True
                    break

            if not matched:
                false_positives += 1

    if len(contradictions) == 0:
        return _clamp(0.5)

    recall = len(found_contradictions) / len(contradictions)
    penalty = false_positives * 0.05
    final = recall - penalty

    return _clamp(final)
