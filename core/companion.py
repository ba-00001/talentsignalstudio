from __future__ import annotations

from core.models import AssessmentResult, CandidateProfile


def companion_response(user_prompt: str, profile: CandidateProfile, assessment: AssessmentResult) -> str:
    prompt = (user_prompt or "").lower()
    top_gaps = [gap.skill.replace("_", " ") for gap in assessment.gaps[:3]]
    gap_text = ", ".join(top_gaps) if top_gaps else "no critical gaps"

    if "improve" in prompt or "better" in prompt:
        return (
            f"Focus on {gap_text}. Start with 2-week actions in the upskilling plan, "
            "then rerun the simulator with a +10 to +15 point uplift on those skills."
        )
    if "role" in prompt or "job" in prompt:
        best = assessment.recommendations[0] if assessment.recommendations else (profile.target_role, 0.0)
        return (
            f"Best current role fit is {best[0]} at {best[1]:.1f}%. "
            "Use the Jobs and Training section to open live listings and matched learning pathways."
        )
    if "interview" in prompt:
        return (
            "Use the Behavioral set for storytelling and ownership examples, and the Technical set for "
            "structured problem-solving plus AI output verification."
        )
    return (
        f"Current fit is {assessment.fit_score:.1f}% ({assessment.fit_band}) with readiness likelihood "
        f"{assessment.quant_likelihood_pct:.1f}%. Highest-priority development areas: {gap_text}."
    )
