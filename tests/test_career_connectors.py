from __future__ import annotations

from core.career_connectors import build_job_search_links, build_training_links


def test_job_search_links_include_platforms():
    links = build_job_search_links("Analyst", "New York")
    sources = {item["source"] for item in links}
    assert "LinkedIn" in sources
    assert "Indeed" in sources


def test_training_links_have_bootcamp_connector():
    links = build_training_links(["ai literacy", "communication"])
    providers = {item["provider"] for item in links}
    assert "General Assembly" in providers
