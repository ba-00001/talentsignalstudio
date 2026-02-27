from __future__ import annotations

import math
from statistics import pstdev

from core.models import AssessmentResult, CandidateProfile, RoleBenchmark, SkillGap
from core.recommendations import (
    build_interview_pack,
    build_upskilling_plan,
    recommend_roles,
)

DIMENSION_WEIGHTS = {
    "skill_alignment": 0.40,
    "tool_collaboration": 0.20,
    "applied_evidence": 0.20,
    "communication_reasoning": 0.20,
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _fit_band(score_pct: float) -> str:
    if score_pct < 40:
        return "Low"
    if score_pct < 70:
        return "Emerging"
    if score_pct < 85:
        return "Strong"
    return "High"


def _skill_alignment(profile: CandidateProfile, benchmark: RoleBenchmark) -> tuple[float, list[SkillGap]]:
    gaps: list[SkillGap] = []
    ratios: list[float] = []
    for skill, expected in benchmark.required_skills.items():
        current = _clamp(profile.skills.get(skill, 0.0))
        expected = _clamp(expected)
        weight = benchmark.weight_map.get(skill, 1.0)
        ratio = 1.0 if expected <= 0 else _clamp(current / expected)
        ratios.append(ratio * weight)
        delta = _clamp(expected - current)
        gaps.append(
            SkillGap(
                skill=skill,
                expected_level=expected,
                current_level=current,
                delta=delta,
                impact=delta * weight,
            )
        )

    denom = sum(benchmark.weight_map.get(skill, 1.0) for skill in benchmark.required_skills) or 1.0
    alignment = _clamp(sum(ratios) / denom)
    gaps.sort(key=lambda g: g.impact, reverse=True)
    return alignment, gaps


def _tool_collaboration(profile: CandidateProfile) -> float:
    ai_lit = profile.skills.get("ai_literacy", 0.0)
    tool_fluency = profile.skills.get("tool_fluency", 0.0)
    usage = _clamp(profile.tool_usage_frequency)
    return _clamp(0.4 * usage + 0.3 * ai_lit + 0.3 * tool_fluency)


def _applied_evidence(profile: CandidateProfile) -> float:
    projects_norm = _clamp(profile.projects_count / 6.0)
    internship_norm = _clamp(profile.internship_months / 12.0)
    exp_norm = _clamp(profile.experience_months / 24.0)
    return _clamp(0.4 * projects_norm + 0.4 * internship_norm + 0.2 * exp_norm)


def _communication_reasoning(profile: CandidateProfile) -> float:
    comm = profile.skills.get("communication", 0.0)
    reasoning = profile.skills.get("critical_thinking", 0.0)
    self_rate = _clamp(profile.self_proficiency)
    return _clamp(0.4 * comm + 0.4 * reasoning + 0.2 * self_rate)


def _missing_data_penalty(profile: CandidateProfile) -> float:
    required_keys = {
        "ai_literacy",
        "analytical_thinking",
        "communication",
        "domain_knowledge",
        "tool_fluency",
        "critical_thinking",
    }
    missing = sum(1 for key in required_keys if key not in profile.skills)
    return min(0.08, missing * 0.0125)


def _compute_quant_likelihood(fit_score: float, profile: CandidateProfile) -> tuple[float, str]:
    evidence_values = [
        _clamp(profile.projects_count / 6.0),
        _clamp(profile.internship_months / 12.0),
        _clamp(profile.experience_months / 24.0),
    ]
    variance_penalty = min(0.08, pstdev(evidence_values) * 0.20)
    z = (fit_score - 0.58) * 7.0 - variance_penalty * 6.0
    prob = 1.0 / (1.0 + math.exp(-z))
    likelihood = round(_clamp(prob) * 100.0, 1)
    if variance_penalty > 0.06:
        confidence = "Low"
    elif variance_penalty > 0.03:
        confidence = "Medium"
    else:
        confidence = "High"
    return likelihood, confidence


def apply_interview_self_score(likelihood_pct: float, interview_score: float | None) -> float:
    if interview_score is None:
        return likelihood_pct
    adjustment = (interview_score - 50.0) * 0.18
    return round(max(0.0, min(100.0, likelihood_pct + adjustment)), 1)


def score_candidate(
    profile: CandidateProfile,
    benchmark: RoleBenchmark,
    role_catalog: list[RoleBenchmark] | None = None,
) -> AssessmentResult:
    skill_alignment, gaps = _skill_alignment(profile, benchmark)
    tool = _tool_collaboration(profile)
    evidence = _applied_evidence(profile)
    comm_reason = _communication_reasoning(profile)

    weighted = (
        DIMENSION_WEIGHTS["skill_alignment"] * skill_alignment
        + DIMENSION_WEIGHTS["tool_collaboration"] * tool
        + DIMENSION_WEIGHTS["applied_evidence"] * evidence
        + DIMENSION_WEIGHTS["communication_reasoning"] * comm_reason
    )
    weighted = _clamp(weighted - _missing_data_penalty(profile))
    fit_score_pct = round(weighted * 100.0, 1)
    fit_band = _fit_band(fit_score_pct)

    quant_likelihood, confidence = _compute_quant_likelihood(weighted, profile)
    quant_likelihood = apply_interview_self_score(quant_likelihood, profile.interview_self_score)

    role_recos = recommend_roles(profile, role_catalog or [benchmark])
    upskilling = build_upskilling_plan(gaps[:5])
    interview_pack = build_interview_pack(gaps[:3])

    return AssessmentResult(
        fit_score=fit_score_pct,
        fit_band=fit_band,
        gaps=gaps,
        quant_likelihood_pct=quant_likelihood,
        confidence=confidence,
        dimension_scores={
            "skill_alignment": round(skill_alignment * 100.0, 1),
            "tool_collaboration": round(tool * 100.0, 1),
            "applied_evidence": round(evidence * 100.0, 1),
            "communication_reasoning": round(comm_reason * 100.0, 1),
        },
        recommendations=role_recos,
        upskilling_plan=upskilling,
        interview_pack=interview_pack,
    )
