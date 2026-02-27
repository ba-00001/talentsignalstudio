from __future__ import annotations

import json
from pathlib import Path

from core.models import CandidateProfile, RoleBenchmark
from core.scoring import score_candidate

BASE_DIR = Path(__file__).resolve().parents[1]


def _load_benchmarks() -> list[RoleBenchmark]:
    with (BASE_DIR / "data" / "role_benchmarks.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return [
        RoleBenchmark(
            role=item["role"],
            required_skills=item["required_skills"],
            preferred_skills=item.get("preferred_skills", {}),
            weight_map=item.get("weight_map", {}),
        )
        for item in raw
    ]


def _profile(skills: dict[str, float], projects: int, intern: int, exp: int) -> CandidateProfile:
    return CandidateProfile(
        name="Test Candidate",
        target_role="Analyst",
        education_level="Bachelor",
        experience_months=exp,
        self_proficiency=0.6,
        tool_usage_frequency=0.6,
        projects_count=projects,
        internship_months=intern,
        skills=skills,
    )


def test_score_monotonicity():
    role_catalog = _load_benchmarks()
    analyst = [r for r in role_catalog if r.role == "Analyst"][0]
    weaker = _profile(
        {
            "ai_literacy": 0.4,
            "analytical_thinking": 0.4,
            "communication": 0.5,
            "domain_knowledge": 0.4,
            "tool_fluency": 0.4,
            "critical_thinking": 0.5,
        },
        projects=1,
        intern=1,
        exp=2,
    )
    stronger = _profile(
        {
            "ai_literacy": 0.8,
            "analytical_thinking": 0.8,
            "communication": 0.8,
            "domain_knowledge": 0.7,
            "tool_fluency": 0.8,
            "critical_thinking": 0.8,
        },
        projects=4,
        intern=6,
        exp=12,
    )
    weak_score = score_candidate(weaker, analyst, role_catalog).fit_score
    strong_score = score_candidate(stronger, analyst, role_catalog).fit_score
    assert strong_score >= weak_score


def test_gap_ranking_is_deterministic():
    role_catalog = _load_benchmarks()
    analyst = [r for r in role_catalog if r.role == "Analyst"][0]
    profile = _profile(
        {
            "ai_literacy": 0.6,
            "analytical_thinking": 0.4,
            "communication": 0.7,
            "domain_knowledge": 0.5,
            "tool_fluency": 0.6,
            "critical_thinking": 0.55,
        },
        projects=2,
        intern=3,
        exp=6,
    )
    result_a = score_candidate(profile, analyst, role_catalog)
    result_b = score_candidate(profile, analyst, role_catalog)
    assert [g.skill for g in result_a.gaps] == [g.skill for g in result_b.gaps]


def test_end_to_end_contains_all_artifacts():
    role_catalog = _load_benchmarks()
    analyst = [r for r in role_catalog if r.role == "Analyst"][0]
    profile = _profile(
        {
            "ai_literacy": 0.7,
            "analytical_thinking": 0.7,
            "communication": 0.7,
            "domain_knowledge": 0.6,
            "tool_fluency": 0.7,
            "critical_thinking": 0.7,
        },
        projects=3,
        intern=4,
        exp=10,
    )
    result = score_candidate(profile, analyst, role_catalog)
    assert isinstance(result.fit_score, float)
    assert result.gaps
    assert len(result.recommendations) == 3
    assert set(result.upskilling_plan.keys()) == {"2-week", "30-day", "90-day"}
    assert len(result.interview_pack) == 8
    categories = {item.get("category") for item in result.interview_pack}
    assert categories == {"Behavioral", "Technical"}
