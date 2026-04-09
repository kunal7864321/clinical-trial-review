"""Graders for clinical trial review tasks.

Every grader returns a score strictly in (0, 1) — never 0.0, never 1.0.
"""

# Strict bounds — scores must be > 0 and < 1
_SCORE_MIN = 0.000001
_SCORE_MAX = 0.99


def _clamp(value: float) -> float:
    """Clamp a score to be strictly within (0, 1)."""
    clamped = max(_SCORE_MIN, min(_SCORE_MAX, value))
    return round(float(clamped), 6)


def grade_task1(agent_actions, ground_truth):
    """
    Missing section detection — score strictly in (0, 1).
    Task: Identify sections that are present in REQUIRED_SECTIONS but missing from the protocol.
    """
    actual_missing = set(ground_truth.get("missing_sections", []))

    if not actual_missing:
        # If nothing is missing, any "flag_issue" is a false positive
        # But we give a high base score if the agent did nothing wrong
        if not agent_actions:
            return _clamp(0.95)
        return _clamp(0.5)

    correctly_flagged = set()
    false_positives = 0

    for action in agent_actions:
        if action.get("action_type") == "flag_issue":
            section = action.get("target_section")
            if section in actual_missing:
                correctly_flagged.add(section)
            else:
                false_positives += 1

    # Precision/Recall based scoring
    recall = len(correctly_flagged) / len(actual_missing)
    penalty = false_positives * 0.1
    final_score = recall - penalty

    return _clamp(final_score)


def grade_task2(agent_actions, ground_truth):
    """
    Dosage safety compliance — score strictly in (0, 1).
    Task: Identify drug dosages that exceed the maximum allowed limits.
    """
    unsafe_dosages = ground_truth.get("unsafe_dosages", [])
    
    if not unsafe_dosages:
        if not agent_actions:
            return _clamp(0.95)
        return _clamp(0.5)

    unsafe_drugs = [u["drug"] for u in unsafe_dosages]
    correctly_flagged = set()
    false_positives = 0

    for action in agent_actions:
        if action.get("action_type") == "flag_issue":
            desc = action.get("issue_description", "").lower()
            target = action.get("target_section", "").lower()
            
            matched = False
            for drug in unsafe_drugs:
                # Check if the drug name appears in description or target section
                if drug.lower() in desc or drug.lower() in target:
                    correctly_flagged.add(drug)
                    matched = True
                    break
            
            if not matched:
                false_positives += 1

    recall = len(correctly_flagged) / len(unsafe_drugs)
    penalty = false_positives * 0.1
    final_score = recall - penalty

    return _clamp(final_score)


def grade_task3(agent_actions, ground_truth):
    """
    Contradiction detection — score strictly in (0, 1).
    Task: Find internal contradictions between different sections of the protocol.
    """
    contradictions = ground_truth.get("contradictions", [])

    if not contradictions:
        if not agent_actions:
            return _clamp(0.95)
        return _clamp(0.5)

    found_indices = set()
    false_positives = 0

    for action in agent_actions:
        if action.get("action_type") == "flag_issue":
            desc = action.get("issue_description", "").lower()
            target = action.get("target_section", "").lower()
            matched = False

            for i, contradiction in enumerate(contradictions):
                sec_a = contradiction["section_a"].lower()
                sec_b = contradiction["section_b"].lower()
                
                # Check if both sections involved in the contradiction are mentioned
                mentions_sections = (sec_a in desc or sec_a in target) and (sec_b in desc or sec_b in target)
                
                # Also check for key thematic words from the ground truth description
                desc_words = contradiction["description"].lower().split()
                key_words = [w for w in desc_words if len(w) > 5][:5]
                keyword_matches = sum(1 for w in key_words if w in desc)

                if mentions_sections or keyword_matches >= 2:
                    found_indices.add(i)
                    matched = True
                    break

            if not matched:
                false_positives += 1

    recall = len(found_indices) / len(contradictions)
    penalty = false_positives * 0.1
    final_score = recall - penalty

    return _clamp(final_score)
