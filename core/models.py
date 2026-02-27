from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SkillGap:
    skill: str
    expected_level: float
    current_level: float
    delta: float
    impact: float


@dataclass
class CandidateProfile:
    name: str
    target_role: str
    education_level: str
    experience_months: int
    self_proficiency: float
    tool_usage_frequency: float
    projects_count: int
    internship_months: int
    skills: dict[str, float]
    interview_self_score: float | None = None


@dataclass
class RoleBenchmark:
    role: str
    required_skills: dict[str, float]
    preferred_skills: dict[str, float] = field(default_factory=dict)
    weight_map: dict[str, float] = field(default_factory=dict)


@dataclass
class AssessmentResult:
    fit_score: float
    fit_band: str
    gaps: list[SkillGap]
    quant_likelihood_pct: float
    confidence: str
    dimension_scores: dict[str, float]
    recommendations: list[tuple[str, float]]
    upskilling_plan: dict[str, list[str]]
    interview_pack: list[dict[str, str]]
