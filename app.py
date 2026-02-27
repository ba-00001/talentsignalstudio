from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from core.career_connectors import build_job_search_links, build_training_links, fetch_live_jobs
from core.companion import companion_response
from core.models import CandidateProfile, RoleBenchmark
from core.parsers import extract_candidate_signals
from core.scoring import score_candidate

APP_TITLE = "TalentSignal Studio"
APP_SUBTITLE = "Make AI-era readiness visible"
DEMO_USERNAME = "demo"
DEMO_PASSWORD = "demo"
ROOT_DIR = Path(__file__).resolve().parent
DEMO_RESUME_PATH = ROOT_DIR / "data" / "demo_resume_fiu_business_analytics.txt"
SKILL_ORDER = [
    "ai_literacy",
    "analytical_thinking",
    "communication",
    "domain_knowledge",
    "tool_fluency",
    "critical_thinking",
]
SCENARIO_PRESETS = {
    "Samantha - Recent Graduate": {
        "name": "Samantha",
        "target_role": "Analyst",
        "education_level": "Bachelor",
        "experience_months": 6,
        "projects_count": 2,
        "self_proficiency": 0.62,
        "tool_usage_frequency": 0.68,
        "internship_months": 3,
        "preferred_location": "United States",
        "skills": {
            "ai_literacy": 0.70,
            "analytical_thinking": 0.64,
            "communication": 0.72,
            "domain_knowledge": 0.55,
            "tool_fluency": 0.68,
            "critical_thinking": 0.62,
        },
    },
    "Daniel - Career Switcher": {
        "name": "Daniel",
        "target_role": "Operations Associate",
        "education_level": "Master",
        "experience_months": 10,
        "projects_count": 3,
        "self_proficiency": 0.58,
        "tool_usage_frequency": 0.60,
        "internship_months": 1,
        "preferred_location": "United States",
        "skills": {
            "ai_literacy": 0.58,
            "analytical_thinking": 0.67,
            "communication": 0.66,
            "domain_knowledge": 0.63,
            "tool_fluency": 0.56,
            "critical_thinking": 0.64,
        },
    },
    "High-Potential Candidate": {
        "name": "Alex",
        "target_role": "Product Analyst",
        "education_level": "Bachelor",
        "experience_months": 18,
        "projects_count": 5,
        "self_proficiency": 0.78,
        "tool_usage_frequency": 0.82,
        "internship_months": 6,
        "preferred_location": "United States",
        "skills": {
            "ai_literacy": 0.82,
            "analytical_thinking": 0.84,
            "communication": 0.76,
            "domain_knowledge": 0.74,
            "tool_fluency": 0.83,
            "critical_thinking": 0.81,
        },
    },
    "FIU Business Analytics Demo": {
        "name": "Camila Perez",
        "target_role": "Analyst",
        "education_level": "Bachelor",
        "experience_months": 8,
        "projects_count": 4,
        "self_proficiency": 0.74,
        "tool_usage_frequency": 0.80,
        "internship_months": 4,
        "preferred_location": "Miami, FL",
        "skills": {
            "ai_literacy": 0.80,
            "analytical_thinking": 0.82,
            "communication": 0.78,
            "domain_knowledge": 0.72,
            "tool_fluency": 0.81,
            "critical_thinking": 0.79,
        },
    },
}


def load_taxonomy() -> list[dict[str, str]]:
    with (ROOT_DIR / "data" / "skill_taxonomy.json").open("r", encoding="utf-8") as f:
        return json.load(f)["skills"]


def load_benchmarks() -> list[RoleBenchmark]:
    with (ROOT_DIR / "data" / "role_benchmarks.json").open("r", encoding="utf-8") as f:
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


def role_lookup(benchmarks: list[RoleBenchmark]) -> dict[str, RoleBenchmark]:
    return {benchmark.role: benchmark for benchmark in benchmarks}


def as_pct_label(value: float) -> str:
    return f"{value:.1f}%"


def default_form_values() -> dict:
    return {
        "name": "Samantha Candidate",
        "target_role": "Analyst",
        "education_level": "Bachelor",
        "experience_months": 6,
        "projects_count": 2,
        "self_proficiency": 0.60,
        "tool_usage_frequency": 0.65,
        "internship_months": 3,
        "preferred_location": "United States",
        "skills": {key: 0.50 for key in SKILL_ORDER},
    }


def ensure_state():
    if "assessment" not in st.session_state:
        st.session_state["assessment"] = None
    if "last_profile" not in st.session_state:
        st.session_state["last_profile"] = None
    if "form_defaults" not in st.session_state:
        st.session_state["form_defaults"] = default_form_values()
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False


def inject_styles():
    st.markdown(
        """
        <style>
        .hero-wrap {
            background: radial-gradient(circle at 20% 20%, #1ed760 0%, #10954b 35%, #0f172a 100%);
            border-radius: 18px;
            padding: 28px;
            color: #f8fff7;
            margin-bottom: 18px;
        }
        .hero-title { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.4rem; }
        .hero-sub { opacity: 0.92; font-size: 1.03rem; }
        .login-card {
            background: #0f172a;
            border: 1px solid #1f2937;
            border-radius: 16px;
            padding: 20px;
            color: #e5e7eb;
        }
        .toolprint-title { margin-top: 0.5rem; font-weight: 700; font-size: 1.2rem; }
        .login-shell { max-width: 460px; margin: 0 auto; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def try_login(username: str, password: str) -> bool:
    return username.strip().lower() == DEMO_USERNAME and password == DEMO_PASSWORD


def apply_preset(preset_name: str):
    st.session_state["form_defaults"] = SCENARIO_PRESETS[preset_name]


def export_payload(profile: CandidateProfile, assessment) -> dict:
    return {
        "candidate": {
            "name": profile.name,
            "target_role": profile.target_role,
            "education_level": profile.education_level,
            "experience_months": profile.experience_months,
            "projects_count": profile.projects_count,
            "internship_months": profile.internship_months,
            "preferred_location": st.session_state.get("preferred_location", "United States"),
            "skills": profile.skills,
        },
        "assessment": {
            "fit_score": assessment.fit_score,
            "fit_band": assessment.fit_band,
            "quant_likelihood_pct": assessment.quant_likelihood_pct,
            "confidence": assessment.confidence,
            "dimension_scores": assessment.dimension_scores,
            "recommendations": assessment.recommendations,
            "upskilling_plan": assessment.upskilling_plan,
            "interview_pack": assessment.interview_pack,
            "gaps": [
                {
                    "skill": g.skill,
                    "expected_level": g.expected_level,
                    "current_level": g.current_level,
                    "delta": g.delta,
                    "impact": g.impact,
                }
                for g in assessment.gaps
            ],
        },
    }


def render_landing_page():
    st.markdown(
        """
        <div class="hero-wrap">
          <div class="hero-title">TalentSignal Studio</div>
          <div class="hero-sub">Professional readiness intelligence for AI-era entry-level careers.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Assessment Lens", "6 Signals", "Fit + Gaps + Role Paths")
    col2.metric("Interview Pack", "8 Questions", "Behavioral + Technical")
    col3.metric("Output Speed", "< 5 min", "Presentation-ready insights")

    center_left, center_mid, center_right = st.columns([1, 1.25, 1])
    with center_mid:
        st.markdown('<div class="login-shell">', unsafe_allow_html=True)
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown("#### Sign In")
        st.caption("Demo credentials: `demo` / `demo`")
        if st.session_state["authenticated"]:
            st.success("Signed in as demo. Use the sidebar menu to open Assessment Workspace.")
        else:
            with st.form("landing_login_form"):
                user_in = st.text_input("Username", value="demo", key="landing_user")
                pass_in = st.text_input("Password", type="password", value="demo", key="landing_pass")
                if st.form_submit_button("Sign In"):
                    if try_login(user_in, pass_in):
                        st.session_state["authenticated"] = True
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="toolprint-title">Toolprint</div>', unsafe_allow_html=True)

    toolprint = pd.DataFrame(
        [
            {"Module": "Candidate Intake", "Capability": "Upload resume, edit skill map, and apply profile presets"},
            {"Module": "Readiness Dashboard", "Capability": "View fit band, quant likelihood, confidence, and gap impact"},
            {"Module": "Role Intelligence", "Capability": "Compare role matches and benchmark skill expectations"},
            {"Module": "Upskilling Planner", "Capability": "Get 2-week, 30-day, and 90-day action roadmap"},
            {"Module": "Interview Studio", "Capability": "Practice behavioral + technical mock interview questions"},
            {"Module": "Career Connectors", "Capability": "Open LinkedIn/Indeed jobs and training/bootcamp pathways"},
            {"Module": "AI Companion", "Capability": "Ask guidance questions on roles, improvement, and interviews"},
            {"Module": "Export", "Capability": "Download complete JSON report for presentations and grading"},
        ]
    )
    st.dataframe(toolprint, hide_index=True, use_container_width=True)
    st.markdown("### Demo Assets")
    st.write("- Profile: FIU Business student studying Business Analytics")
    st.write("- After login, use sidebar preset: `FIU Business Analytics Demo`")
    if DEMO_RESUME_PATH.exists():
        st.download_button(
            "Download Dummy Resume (FIU Demo)",
            data=DEMO_RESUME_PATH.read_text(encoding="utf-8"),
            file_name="FIU_Business_Analytics_Demo_Resume.txt",
            mime="text/plain",
        )


def render_assessment_workspace(skills_meta, skill_labels, benchmarks, benchmarks_by_role, roles):
    defaults = st.session_state["form_defaults"]
    with st.expander("Section A - Candidate Intake", expanded=True):
        with st.form("intake_form"):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Candidate name", value=defaults["name"])
                target_role = st.selectbox("Target role", roles, index=roles.index(defaults["target_role"]))
                education_level = st.selectbox(
                    "Education level",
                    ["High School", "Associate", "Bachelor", "Master", "Other"],
                    index=["High School", "Associate", "Bachelor", "Master", "Other"].index(defaults["education_level"]),
                )
                experience_months = st.number_input("Experience (months)", min_value=0, max_value=120, value=int(defaults["experience_months"]))
                projects_count = st.number_input("Projects completed", min_value=0, max_value=50, value=int(defaults["projects_count"]))
            with c2:
                self_proficiency = st.slider("Self-rated proficiency", 0, 100, int(defaults["self_proficiency"] * 100)) / 100.0
                tool_usage_frequency = st.slider("AI/tool usage frequency", 0, 100, int(defaults["tool_usage_frequency"] * 100)) / 100.0
                internship_months = st.number_input("Internship months", min_value=0, max_value=60, value=int(defaults["internship_months"]))
                preferred_location = st.text_input("Preferred job location", value=defaults.get("preferred_location", "United States"))
                uploaded_file = st.file_uploader("Resume upload (optional)", type=["pdf", "docx", "txt"])

            parsed_signals = extract_candidate_signals(uploaded_file) if uploaded_file else {"skills": {}, "keywords": []}
            parsed_skill_scores = parsed_signals.get("skills", {}) if isinstance(parsed_signals, dict) else {}
            st.markdown("Manual skills entry (takes precedence over resume parsing)")
            selected_skills = st.multiselect(
                "Select skills to set manually",
                options=[item["key"] for item in skills_meta],
                default=[item["key"] for item in skills_meta],
                format_func=lambda key: skill_labels[key],
            )

            skill_values: dict[str, float] = {}
            cols = st.columns(3)
            for i, skill_key in enumerate([item["key"] for item in skills_meta]):
                with cols[i % 3]:
                    fallback = float(defaults["skills"].get(skill_key, 0.50))
                    default_value = float(parsed_skill_scores.get(skill_key, fallback))
                    default_pct = int(round(default_value * 100))
                    if skill_key in selected_skills:
                        val = st.slider(skill_labels[skill_key], 0, 100, default_pct, key=f"manual_{skill_key}") / 100.0
                        skill_values[skill_key] = val
                    else:
                        skill_values[skill_key] = float(parsed_skill_scores.get(skill_key, fallback))
                        st.caption(f"{skill_labels[skill_key]} from resume parse: {as_pct_label(skill_values[skill_key] * 100)}")

            submitted = st.form_submit_button("Run readiness assessment")

    if submitted:
        profile = CandidateProfile(
            name=name,
            target_role=target_role,
            education_level=education_level,
            experience_months=int(experience_months),
            self_proficiency=self_proficiency,
            tool_usage_frequency=tool_usage_frequency,
            projects_count=int(projects_count),
            internship_months=int(internship_months),
            skills=skill_values,
        )
        st.session_state["form_defaults"] = {
            "name": name,
            "target_role": target_role,
            "education_level": education_level,
            "experience_months": int(experience_months),
            "projects_count": int(projects_count),
            "self_proficiency": self_proficiency,
            "tool_usage_frequency": tool_usage_frequency,
            "internship_months": int(internship_months),
            "preferred_location": preferred_location,
            "skills": skill_values,
        }
        st.session_state["preferred_location"] = preferred_location
        assessment = score_candidate(profile, benchmarks_by_role[target_role], benchmarks)
        st.session_state["assessment"] = assessment
        st.session_state["last_profile"] = profile

    assessment = st.session_state.get("assessment")
    profile = st.session_state.get("last_profile")
    if not (assessment and profile):
        st.info("Complete Section A and run readiness assessment to view results.")
        return

    with st.expander("Section B - Readiness Dashboard", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Fit Score", as_pct_label(assessment.fit_score), assessment.fit_band)
        c1.progress(int(assessment.fit_score) / 100.0)
        c2.metric("Quant Readiness Likelihood", as_pct_label(assessment.quant_likelihood_pct), f"Confidence: {assessment.confidence}")
        c3.markdown("Dimension Scores")
        for key, value in assessment.dimension_scores.items():
            c3.write(f"- {key.replace('_', ' ').title()}: {as_pct_label(value)}")

        benchmark = benchmarks_by_role[profile.target_role]
        compare_df = pd.DataFrame(
            [
                {
                    "Skill": skill_labels.get(skill, skill),
                    "Candidate": round(profile.skills.get(skill, 0.0) * 100, 1),
                    "Benchmark": round(expected * 100, 1),
                    "Gap": round(max(0.0, (expected - profile.skills.get(skill, 0.0)) * 100), 1),
                }
                for skill, expected in benchmark.required_skills.items()
            ]
        )
        st.markdown("Skill vs Target Benchmark")
        st.bar_chart(compare_df.set_index("Skill")[["Candidate", "Benchmark"]])
        st.dataframe(compare_df, use_container_width=True, hide_index=True)

    with st.expander("Section C - Recommendations + What-If Planner", expanded=True):
        role_df = pd.DataFrame([{"Role": role, "Match %": score} for role, score in assessment.recommendations])
        st.dataframe(role_df, use_container_width=True, hide_index=True)
        plan = assessment.upskilling_plan
        a, b, c = st.columns(3)
        a.markdown("**2-week**")
        for item in plan.get("2-week", []):
            a.write(f"- {item}")
        b.markdown("**30-day**")
        for item in plan.get("30-day", []):
            b.write(f"- {item}")
        c.markdown("**90-day**")
        for item in plan.get("90-day", []):
            c.write(f"- {item}")

    with st.expander("Section D - AI Mock Interviews", expanded=True):
        interview_score = st.slider("Optional self-score for mock interview (0-100)", 0, 100, 50)
        if st.button("Apply interview self-score to readiness likelihood"):
            profile.interview_self_score = float(interview_score)
            assessment = score_candidate(profile, benchmarks_by_role[profile.target_role], benchmarks)
            st.session_state["assessment"] = assessment
            st.session_state["last_profile"] = profile
            st.success("Updated readiness likelihood with interview self-score.")

        behavioral = [q for q in assessment.interview_pack if q.get("category") == "Behavioral"]
        technical = [q for q in assessment.interview_pack if q.get("category") == "Technical"]
        x, y = st.columns(2)
        x.markdown("#### Behavioral")
        for item in behavioral:
            x.write(item["question"])
            x.caption(item["rubric"])
        y.markdown("#### Technical")
        for item in technical:
            y.write(item["question"])
            y.caption(item["rubric"])

    with st.expander("Section E - Job Postings + Training Connectors", expanded=True):
        preferred_location = st.session_state.get("preferred_location", "United States")
        st.caption("Live API results require RAPIDAPI_KEY.")
        live_jobs = fetch_live_jobs(profile.target_role, preferred_location, max_results=8)
        if live_jobs:
            for job in live_jobs:
                st.write(f"- [{job['title']} - {job['company']} ({job['source']})]({job['url']})")
        else:
            st.info("No live API results found. Use direct platform links.")
        for item in build_job_search_links(profile.target_role, preferred_location):
            st.write(f"- [{item['title']}]({item['url']})")

        top_gap_labels = [g.skill.replace("_", " ") for g in assessment.gaps[:3]]
        for item in build_training_links(top_gap_labels):
            st.write(f"- [{item['provider']}: {item['title']}]({item['url']})")

    with st.expander("Section F - AI Companion", expanded=True):
        prompt = st.text_input("Ask your AI companion", value="How can I improve my readiness fastest?")
        if st.button("Get Companion Guidance"):
            st.write(companion_response(prompt, profile, assessment))

    with st.expander("Section G - Narrative Mapping + Export", expanded=True):
        mapping = pd.DataFrame(
            [
                {"Feature": "Resume upload + skill mapping", "Narrative Role": "Converts static credentials into dynamic readiness signals"},
                {"Feature": "Fit % + gap analysis", "Narrative Role": "Reduces ambiguity candidates feel about readiness"},
                {"Feature": "Role recommendations", "Narrative Role": "Helps candidates interpret evolving pathways"},
                {"Feature": "Upskilling recommendations", "Narrative Role": "Makes developmental progress visible"},
                {"Feature": "AI mock interviews", "Narrative Role": "Shows capability beyond resume signaling"},
                {"Feature": "Quant readiness likelihood", "Narrative Role": "Supports manager-side evaluation interpretation"},
            ]
        )
        st.dataframe(mapping, hide_index=True, use_container_width=True)
        report = export_payload(profile, assessment)
        st.download_button("Download Assessment JSON", data=json.dumps(report, indent=2), file_name=f"{profile.name.lower().replace(' ', '_')}_assessment.json", mime="application/json")


st.set_page_config(page_title=APP_TITLE, layout="wide")
inject_styles()
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

skills_meta = load_taxonomy()
skill_labels = {item["key"]: item["label"] for item in skills_meta}
benchmarks = load_benchmarks()
benchmarks_by_role = role_lookup(benchmarks)
roles = [b.role for b in benchmarks]
ensure_state()

if st.session_state["authenticated"]:
    with st.sidebar:
        st.markdown("### Session")
        st.success("Signed in as demo")
        if st.button("Sign Out"):
            st.session_state["authenticated"] = False
            st.session_state["assessment"] = None
            st.session_state["last_profile"] = None
            st.session_state["form_defaults"] = default_form_values()
            st.rerun()
        page = st.radio("Go to", ["Landing", "Assessment Workspace"])
        preset = st.selectbox("Load preset profile", list(SCENARIO_PRESETS.keys()))
        if st.button("Apply Preset"):
            apply_preset(preset)
            st.success(f"Loaded preset: {preset}")
else:
    with st.sidebar:
        st.markdown("### Session")
        st.info("Please sign in on the Landing page.")
    page = "Landing"

if page == "Landing":
    render_landing_page()
else:
    if not st.session_state["authenticated"]:
        st.warning("Please sign in with demo credentials in the sidebar to access the workspace.")
    else:
        render_assessment_workspace(skills_meta, skill_labels, benchmarks, benchmarks_by_role, roles)
