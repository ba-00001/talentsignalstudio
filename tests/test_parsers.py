from __future__ import annotations

from io import BytesIO

from core.parsers import extract_candidate_signals


class BadFile:
    name = "broken.pdf"

    def read(self):
        raise ValueError("cannot read")


def test_parser_fallback_on_malformed_file():
    result = extract_candidate_signals(BadFile())
    assert result["skills"] == {}


def test_parser_handles_txt():
    payload = BytesIO(
        b"Used AI and Python for analysis, communication with stakeholders, and evaluation."
    )
    payload.name = "resume.txt"
    result = extract_candidate_signals(payload)
    assert "skills" in result
    assert isinstance(result["skills"], dict)
