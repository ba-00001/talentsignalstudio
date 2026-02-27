from __future__ import annotations

from io import BytesIO


SKILL_KEYWORDS = {
    "ai_literacy": ["ai", "chatgpt", "llm", "prompt", "automation", "copilot"],
    "analytical_thinking": ["analysis", "analytics", "sql", "statistics", "excel", "tableau"],
    "communication": ["presentation", "stakeholder", "report", "writing", "communication"],
    "domain_knowledge": ["industry", "operations", "market", "customer", "business process"],
    "tool_fluency": ["python", "power bi", "jira", "notion", "spreadsheet", "git"],
    "critical_thinking": ["evaluate", "hypothesis", "tradeoff", "decision", "root cause"],
}


def _read_pdf(file_bytes: bytes) -> str:
    try:
        import pdfplumber
    except ImportError:
        return ""

    text = []
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)


def _read_docx(file_bytes: bytes) -> str:
    try:
        from docx import Document
    except ImportError:
        return ""

    doc = Document(BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text)


def _read_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def _extract_skills(text: str) -> dict[str, float]:
    lowered = text.lower()
    scores: dict[str, float] = {}
    for skill, keywords in SKILL_KEYWORDS.items():
        hits = sum(lowered.count(keyword) for keyword in keywords)
        scores[skill] = min(1.0, hits / 5.0)
    return scores


def extract_candidate_signals(file) -> dict[str, float | list[str] | dict[str, float] | str]:
    if file is None:
        return {"skills": {}, "keywords": [], "text_excerpt": ""}

    try:
        file_bytes = file.read()
    except Exception:
        return {"skills": {}, "keywords": [], "text_excerpt": ""}

    filename = (getattr(file, "name", "") or "").lower()
    text = ""
    try:
        if filename.endswith(".pdf"):
            text = _read_pdf(file_bytes)
        elif filename.endswith(".docx"):
            text = _read_docx(file_bytes)
        elif filename.endswith(".txt"):
            text = _read_txt(file_bytes)
        else:
            text = _read_txt(file_bytes)
    except Exception:
        text = ""

    if not text.strip():
        return {"skills": {}, "keywords": [], "text_excerpt": ""}

    skills = _extract_skills(text)
    keywords = sorted(
        {
            kw
            for _, values in SKILL_KEYWORDS.items()
            for kw in values
            if kw in text.lower()
        }
    )
    return {
        "skills": skills,
        "keywords": keywords[:15],
        "text_excerpt": text[:350],
    }
