def grade_task1(agent_actions, ground_truth):
    """Missing section detection — score 0.0 to 1.0"""
    actual_missing = set(ground_truth["missing_sections"])

    if not actual_missing:
        return 1.0

    correctly_flagged = set()
    false_positives = 0

    for action in agent_actions:
        if action["action_type"] == "flag_issue":
            section = action["target_section"]
            if section in actual_missing:
                correctly_flagged.add(section)
            else:
                false_positives += 1

    precision_score = len(correctly_flagged) / len(actual_missing)
    penalty = false_positives * 0.1
    final = precision_score - penalty

    return round(max(0.0, min(1.0, final)), 3)


def grade_task2(agent_actions, ground_truth):
    """Dosage safety compliance — score 0.0 to 1.0"""
    unsafe_dosages = ground_truth["unsafe_dosages"]

    if not unsafe_dosages:
        return 1.0

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

    recall = len(correctly_flagged) / len(unsafe_drugs)
    penalty = false_positives * 0.1
    final = recall - penalty

    return round(max(0.0, min(1.0, final)), 3)


def grade_task3(agent_actions, ground_truth):
    """Contradiction detection — score 0.0 to 1.0"""
    contradictions = ground_truth["contradictions"]

    if not contradictions:
        return 1.0

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
                sections_mentioned = (sec_a in desc or sec_a in target) and \
                                     (sec_b in desc or sec_b in target)

                if keyword_matches >= 2 or sections_mentioned:
                    found_contradictions.add(i)
                    matched = True
                    break

            if not matched:
                false_positives += 1

    recall = len(found_contradictions) / len(contradictions)
    penalty = false_positives * 0.1
    final = recall - penalty

    return round(max(0.0, min(1.0, final)), 3)