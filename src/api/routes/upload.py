"""
routes/upload.py — POST /upload

Handles resume file upload, multi-strategy text extraction,
NER-based structured data extraction, validation, and MongoDB storage.
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from config import config
from src.db.mongo import resumes_collection
from src.parsers import extract_text_from_pdf, extract_text_from_docx, extract_resume_data
from src.utils.serializers import serialize_document
from src.utils.validators import validate_resume_content

logger = logging.getLogger(__name__)
upload_bp = Blueprint("upload", __name__)

# Ensure upload folder exists at import time
config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


def _allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


@upload_bp.route("/upload", methods=["POST"])
def upload_resume():
    """
    POST /upload
    Accepts a multipart form with a 'file' field (PDF or DOCX).

    Returns:
        200 — { message, user_id, extracted_data, parser_method, pages }
        400 — validation / extraction error
        500 — unexpected server error
    """
    # ── Basic file checks ──────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "error": "Invalid file type",
            "message": "Only PDF and DOCX files are accepted",
        }), 400

    filepath: Path | None = None
    try:
        # ── Save to disk ───────────────────────────────────────────────
        filename = secure_filename(file.filename)
        filepath = config.UPLOAD_FOLDER / filename
        file.save(str(filepath))

        # ── Extract text (upgraded parsers) ───────────────────────────
        ext = filename.rsplit(".", 1)[1].lower()
        if ext == "pdf":
            parse_result = extract_text_from_pdf(str(filepath))
        else:
            parse_result = extract_text_from_docx(str(filepath))

        raw_text: str = parse_result.get("text", "").strip()
        parser_method: str = parse_result.get("method", "unknown")
        pages: int = parse_result.get("pages", 0)

        logger.info(
            "File parsed — method: %s, chars: %d, pages: %d",
            parser_method, len(raw_text), pages,
        )

        if not raw_text:
            return jsonify({
                "error": "Could not extract text from file",
                "message": (
                    "The file appears to be empty or image-only (scanned). "
                    "Please upload a text-based PDF or DOCX."
                ),
                "parser_method": parser_method,
            }), 400

        # ── Structured NER extraction ──────────────────────────────────
        resume_data = extract_resume_data(raw_text)

        # ── Content validation ─────────────────────────────────────────
        validation = validate_resume_content(resume_data)
        if not validation["is_valid"]:
            return jsonify({
                "error": "Invalid resume content",
                "message": "Please upload a valid resume file",
                "details": {
                    "missing_requirements": validation["missing_requirements"],
                    "help": (
                        "Ensure your resume includes: email address, "
                        "at least 2 recognisable skills, and sufficient text."
                    ),
                },
            }), 400

        # ── Persist to MongoDB ─────────────────────────────────────────
        user_id = str(uuid.uuid4())[:8]
        resume_data.update({
            "user_id": user_id,
            "filename": filename,
            "parser_method": parser_method,
            "pages": pages,
            "upload_timestamp": datetime.now().isoformat(),
        })

        resumes_collection.insert_one(resume_data)
        logger.info("Resume stored — user_id: %s", user_id)

        return jsonify({
            "message": "Resume processed successfully",
            "user_id": user_id,
            "parser_method": parser_method,
            "pages": pages,
            "extracted_data": serialize_document(resume_data),
        }), 200

    except Exception as exc:
        logger.exception("Upload error: %s", exc)
        return jsonify({"error": "Processing error", "message": str(exc)}), 500

    finally:
        # Always clean up the uploaded file
        if filepath and filepath.exists():
            try:
                os.remove(filepath)
            except OSError:
                pass
