from __future__ import annotations

import os
from urllib.parse import quote_plus

import requests


def build_job_search_links(role: str, location: str) -> list[dict[str, str]]:
    query = quote_plus(f"{role} entry level")
    loc = quote_plus(location)
    return [
        {
            "source": "LinkedIn",
            "title": f"{role} jobs on LinkedIn",
            "url": f"https://www.linkedin.com/jobs/search/?keywords={query}&location={loc}",
        },
        {
            "source": "Indeed",
            "title": f"{role} jobs on Indeed",
            "url": f"https://www.indeed.com/jobs?q={query}&l={loc}",
        },
        {
            "source": "Google Jobs",
            "title": f"{role} jobs on Google",
            "url": f"https://www.google.com/search?q={query}+jobs+{loc}",
        },
    ]


def fetch_live_jobs(role: str, location: str, max_results: int = 8) -> list[dict[str, str]]:
    api_key = os.getenv("RAPIDAPI_KEY")
    if not api_key:
        return []

    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
    }
    params = {
        "query": f"{role} entry level in {location}",
        "page": "1",
        "num_pages": "1",
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=12)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    jobs = []
    for item in payload.get("data", [])[:max_results]:
        publisher = (item.get("job_publisher") or "").strip()
        apply_link = item.get("job_apply_link") or item.get("job_google_link") or ""
        if not apply_link:
            continue
        jobs.append(
            {
                "source": publisher or "Job Board",
                "title": item.get("job_title", "Job opening"),
                "company": item.get("employer_name", "Unknown"),
                "location": item.get("job_city", "") or item.get("job_country", ""),
                "url": apply_link,
            }
        )
    return jobs


def build_training_links(skills: list[str]) -> list[dict[str, str]]:
    top = skills[:3] if skills else ["ai literacy", "analytical thinking", "communication"]
    query = quote_plus(" ".join(top))
    return [
        {
            "provider": "Coursera",
            "title": "Role-aligned courses",
            "url": f"https://www.coursera.org/search?query={query}",
        },
        {
            "provider": "edX",
            "title": "Professional certificates",
            "url": f"https://www.edx.org/search?q={query}",
        },
        {
            "provider": "Udemy",
            "title": "Hands-on bootcamp-style tracks",
            "url": f"https://www.udemy.com/courses/search/?q={query}",
        },
        {
            "provider": "General Assembly",
            "title": "Bootcamp and short-course pathways",
            "url": "https://generalassemb.ly/",
        },
    ]
