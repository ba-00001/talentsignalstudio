from __future__ import annotations

from typing import Iterable

import numpy as np

from core.models import CandidateProfile, RoleBenchmark, SkillGap


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if norm_product == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / norm_product)


def _candidate_vector(profile: CandidateProfile, skills: Iterable[str]) -> np.ndarray:
    return np.array([profile.skills.get(skill, 0.0) for skill in skills], dtype=float)


def _role_vector(benchmark: RoleBenchmark, skills: Iterable[str]) -> np.ndarray:
    return np.array([benchmark.required_skills.get(skill, 0.0) for skill in skills], dtype=float)


def recommend_roles(
    profile: CandidateProfile, role_catalog: list[RoleBenchmark]
) -> list[tuple[str, float]]:
    skill_union = sorted(
        {
            skill
            for benchmark in role_catalog
            for skill in benchmark.required_skills.keys()
        }
    )
    candidate_vec = _candidate_vector(profile, skill_union)
    results: list[tuple[str, float]] = []
    for benchmark in role_catalog:
        role_vec = _role_vector(benchmark, skill_union)
        sim = _cosine_similarity(candidate_vec, role_vec)
        results.append((benchmark.role, round(max(0.0, min(1.0, sim)) * 100.0, 1)))
    results.sort(key=lambda r: r[1], reverse=True)
    return results[:3]


UPSKILLING_ACTIONS = {
    "ai_literacy": (
        "Practice prompting and validation with two AI tools weekly.",
        "Build one AI-assisted mini project with documented decisions.",
        "Lead a peer walkthrough on responsible AI usage.",
    ),
    "analytical_thinking": (
        "Solve 3 structured case exercises and write short analyses.",
        "Create one KPI dashboard from sample data.",
        "Complete an advanced analytics capstone and present insights.",
    ),
    "communication": (
        "Record and review 3 concise problem-explanation videos.",
        "Write weekly one-page executive summaries on project progress.",
        "Facilitate a mock stakeholder update and collect feedback.",
    ),
    "domain_knowledge": (
        "Map core industry terms and workflows for target role.",
        "Complete a role-relevant business simulation.",
        "Produce a domain playbook with common scenarios and responses.",
    ),
    "tool_fluency": (
        "Automate one repetitive task with spreadsheet/Python workflow.",
        "Ship two portfolio artifacts using role-standard tools.",
        "Demonstrate end-to-end toolchain usage in a timed exercise.",
    ),
    "critical_thinking": (
        "Use claim-evidence-reasoning structure in 5 practice prompts.",
        "Evaluate AI-generated outputs and log failure patterns.",
        "Run decision retrospectives for two project deliverables.",
    ),
}


def build_upskilling_plan(gaps: list[SkillGap]) -> dict[str, list[str]]:
    top = gaps[:3]
    plan = {"2-week": [], "30-day": [], "90-day": []}
    for gap in top:
        actions = UPSKILLING_ACTIONS.get(
            gap.skill,
            (
                f"Do targeted practice for {gap.skill}.",
                f"Build one project artifact focused on {gap.skill}.",
                f"Demonstrate advanced application of {gap.skill}.",
            ),
        )
        plan["2-week"].append(actions[0])
        plan["30-day"].append(actions[1])
        plan["90-day"].append(actions[2])
    return plan


def build_interview_pack(gaps: list[SkillGap]) -> list[dict[str, str]]:
    behavioral_rubric = "Rubric: clarity, ownership, collaboration, ethics/judgment."
    technical_rubric = "Rubric: structured reasoning, tool collaboration, tradeoffs, verification."
    behavioral_by_skill = {
        "ai_literacy": "Describe a time you used AI to accelerate work and how you verified quality.",
        "analytical_thinking": "Tell me about a decision where incomplete data forced you to adapt.",
        "communication": "Give an example of explaining a complex idea to a non-technical stakeholder.",
        "domain_knowledge": "Describe how you learned a new domain quickly to unblock a project.",
        "tool_fluency": "Tell me about a workflow you improved by combining multiple tools.",
        "critical_thinking": "Share a situation where you challenged an assumption and changed direction.",
    }
    technical_by_skill = {
        "ai_literacy": "Design a prompt-validation loop for generating reliable weekly reports.",
        "analytical_thinking": "Walk through diagnosing a KPI drop with only partial funnel data.",
        "communication": "Draft a one-minute executive update from a noisy analysis output.",
        "domain_knowledge": "How would you model operational constraints for this role's priorities?",
        "tool_fluency": "Show how you would combine spreadsheet + Python to automate a recurring task.",
        "critical_thinking": "How do you detect and correct plausible but wrong AI-generated outputs?",
    }
    selected_skills = [gap.skill for gap in gaps] or ["ai_literacy", "communication", "critical_thinking"]
    questions: list[dict[str, str]] = []

    for idx in range(4):
        skill = selected_skills[idx % len(selected_skills)]
        q = behavioral_by_skill.get(skill, f"Tell me how you demonstrate strength in {skill}.")
        questions.append(
            {
                "category": "Behavioral",
                "question": f"B{idx + 1}. {q}",
                "rubric": behavioral_rubric,
            }
        )
    for idx in range(4):
        skill = selected_skills[idx % len(selected_skills)]
        q = technical_by_skill.get(skill, f"How do you apply {skill} in real project execution?")
        questions.append(
            {
                "category": "Technical",
                "question": f"T{idx + 1}. {q}",
                "rubric": technical_rubric,
            }
        )
    return questions
