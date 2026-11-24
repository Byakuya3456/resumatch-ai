"""
validators.py — Resume content validation logic.
"""


def validate_resume_content(resume_data: dict) -> dict:
    """
    Validate that extracted resume data meets minimum requirements.

    Returns:
        {
            "is_valid": bool,
            "missing_requirements": list[str]
        }
    """
    missing: list[str] = []

    # Email is required
    if not resume_data.get("email"):
        missing.append("No valid email address found")

    # At least 2 skills detected
    skills_found = len(resume_data.get("skills", []))
    if skills_found < 2:
        missing.append(
            f"Insufficient skills detected "
            f"(found {skills_found}, need at least 2)"
        )

    # Raw text must be substantial enough to be a real resume
    raw_len = len(resume_data.get("raw_text", ""))
    if raw_len < 100:
        missing.append(
            f"Insufficient resume content (found {raw_len} characters, "
            f"need at least 100)"
        )

    return {
        "is_valid": len(missing) == 0,
        "missing_requirements": missing,
    }
