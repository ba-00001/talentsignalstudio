from __future__ import annotations

import json
from pathlib import Path

from core.models import CandidateProfile, RoleBenchmark
from core.recommendations import recommend_roles

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


def test_recommendation_order_stable():
    catalog = _load_benchmarks()
    profile = CandidateProfile(
        name="Samantha",
        target_role="Analyst",
        education_level="Bachelor",
        experience_months=8,
        self_proficiency=0.7,
        tool_usage_frequency=0.7,
        projects_count=3,
        internship_months=4,
        skills={
            "ai_literacy": 0.75,
            "analytical_thinking": 0.8,
            "communication": 0.7,
            "domain_knowledge": 0.65,
            "tool_fluency": 0.8,
            "critical_thinking": 0.8,
        },
    )
    rec_a = recommend_roles(profile, catalog)
    rec_b = recommend_roles(profile, catalog)
    assert rec_a == rec_b
