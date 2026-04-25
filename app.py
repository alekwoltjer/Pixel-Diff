"""
Blueprint Diff – Flask Web App
==============================
Serves a minimal UI where users upload a Construction Drawing and a Shop
Drawing (PDFs), runs the pixel-diff pipeline, then calls Claude Vision
to produce a structured list of key differences.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
from pathlib import Path

import anthropic
import cv2
import fitz  # PyMuPDF
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from blueprint_diff import generate_blueprint_diff, pdf_to_image

# Load .env manually so it works regardless of cwd or shell environment
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ[_k.strip()] = _v.strip()  # force override empty shell vars

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB total

ALLOWED_MIME = {"application/pdf"}
MAX_FILE_BYTES = 20 * 1_048_576  # 20 MB per file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_upload(file_storage, label: str) -> None:
    """Raise ValueError with a user-facing message if the upload is invalid."""
    if file_storage is None or file_storage.filename == "":
        raise ValueError(f"Missing upload: {label}")
    if file_storage.mimetype not in ALLOWED_MIME:
        raise ValueError(f"{label} must be a PDF file (got {file_storage.mimetype!r})")


def _img_to_b64_png(np_bgr: np.ndarray) -> str:
    """Encode a BGR NumPy image as a base64 PNG string."""
    ok, buf = cv2.imencode(".png", np_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def _make_deletions_only(diff_bgr: np.ndarray) -> np.ndarray:
    """Return a copy of the diff image with green additions removed.

    Pixels that are pure green (additions) are replaced with the faded gray
    background so only the red deletion highlights remain.
    """
    result = diff_bgr.copy()
    # Pure green pixels in BGR: B=0, G=255, R=0
    green_mask = (
        (diff_bgr[:, :, 0] == 0) &
        (diff_bgr[:, :, 1] == 255) &
        (diff_bgr[:, :, 2] == 0)
    )
    result[green_mask] = (217, 217, 217)  # neutral light gray
    return result


def _parse_json(raw: str) -> object:
    """Strip markdown code fences and parse JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def _draw_highlights_on_image(
    base_bgr: np.ndarray,
    deletions_bgr: np.ndarray,
    min_area_px: int = 800,
    max_area_px: int = 40_000,
    min_aspect: float = 0.15,
    dilate_kernel: int = 8,
    dilate_iters: int = 2,
    highlight_color: tuple[int, int, int] = (0, 200, 255),  # amber in BGR
    highlight_alpha: float = 0.45,
    padding: int = 10,
) -> np.ndarray:
    """Find red clusters in *deletions_bgr* and draw semi-transparent
    highlighted regions on *base_bgr*.

    Each qualifying cluster gets a filled, rounded rectangle drawn on an
    overlay layer which is then blended with the original image using
    alpha compositing — giving a highlighter-pen effect.
    """
    hsv = cv2.cvtColor(deletions_bgr, cv2.COLOR_BGR2HSV)
    mask_lo = cv2.inRange(hsv, (0,   120, 120), (10,  255, 255))
    mask_hi = cv2.inRange(hsv, (170, 120, 120), (180, 255, 255))
    red_mask = cv2.bitwise_or(mask_lo, mask_hi)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    dilated = cv2.dilate(red_mask, kernel, iterations=dilate_iters)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result  = base_bgr.copy()
    overlay = base_bgr.copy()
    kept    = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px or area > max_area_px:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = min(w, h) / max(w, h)
        if aspect < min_aspect:
            continue

        # Expand the bounding box by padding, clamped to image bounds
        ih, iw = base_bgr.shape[:2]
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, iw)
        y2 = min(y + h + padding, ih)

        # Filled rectangle on the overlay layer
        cv2.rectangle(overlay, (x1, y1), (x2, y2), highlight_color, -1)
        kept += 1

    # Blend overlay with original: result = alpha*overlay + (1-alpha)*original
    cv2.addWeighted(overlay, highlight_alpha, result, 1 - highlight_alpha, 0, result)

    app.logger.info(f"Drew {kept} highlight(s) on construction drawing ({len(contours)} raw contours).")
    return result


def _locate_dimensions_vlm(b64_v1: str, b64_deletions: str) -> list[dict]:
    """Ask Claude to return the normalized (x, y) centre of each red region
    that is a dimension annotation.

    Returns a list of dicts: [{"x": 0.23, "y": 0.45, "label": "3'-6\""}, ...]
    where x/y are fractions of image width/height (0,0 = top-left).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return []

    client = anthropic.Anthropic(api_key=api_key)

    user_content = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64_v1},
        },
        {"type": "text", "text": "**Image 1: Construction Drawing** — reference for reading dimension values and their positions."},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64_deletions},
        },
        {
            "type": "text",
            "text": (
                "**Image 2: Deletions-Only Diff** — RED regions are elements from the "
                "Construction Drawing that are missing from the Shop Drawing. "
                "Gray is unchanged context.\n\n"
                "For each RED region that is a dimension annotation (a number, fraction, "
                "or measurement like '3\\'-6\"', '24\"', '1/2', 'EQ', etc.), return its "
                "approximate centre position as a fraction of the image size.\n\n"
                "Rules:\n"
                "- Only include regions that are clearly dimension text or dimension lines with values.\n"
                "- Ignore red regions that are structural lines, borders, hatching, or non-dimension text.\n"
                "- x=0 is the left edge, x=1 is the right edge.\n"
                "- y=0 is the top edge, y=1 is the bottom edge.\n\n"
                "Return ONLY a JSON array. Example:\n"
                '[{"x": 0.12, "y": 0.34, "label": "3\'-6\'"}, '
                '{"x": 0.55, "y": 0.78, "label": "24\\""}]\n\n'
                "If no dimension annotations are highlighted in red, return an empty array [].\n"
                "Return only the JSON array — no surrounding text."
            ),
        },
    ]

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": user_content}],
    )

    try:
        result = _parse_json(message.content[0].text)
        if isinstance(result, list):
            return [
                d for d in result
                if isinstance(d, dict) and "x" in d and "y" in d
            ]
    except (json.JSONDecodeError, ValueError):
        pass

    return []


def _markup_pdf_from_locations(
    shop_pdf_path: Path,
    locations: list[dict],
    diff_image_shape: tuple[int, int],
    output_path: Path,
    radius_pts: float = 24.0,
    stroke_color: tuple[float, float, float] = (0.9, 0.35, 0.0),
    stroke_width: float = 2.5,
) -> Path:
    """Draw orange circles on the shop PDF at AI-identified dimension locations.

    Locations are normalised (x, y) fractions; we scale to PDF point space
    using the page dimensions directly, so DPI never enters the equation.
    """
    doc = fitz.open(str(shop_pdf_path))
    page = doc[0]
    pw, ph = page.rect.width, page.rect.height

    app.logger.info(f"Annotating {len(locations)} dimension location(s) on shop drawing.")

    for loc in locations:
        cx_pts = float(loc["x"]) * pw
        cy_pts = float(loc["y"]) * ph
        rect = fitz.Rect(
            cx_pts - radius_pts, cy_pts - radius_pts,
            cx_pts + radius_pts, cy_pts + radius_pts,
        )
        annot = page.add_circle_annot(rect)
        annot.set_colors(stroke=stroke_color)
        annot.set_border(width=stroke_width)
        if loc.get("label"):
            annot.set_info(content=loc["label"])
        annot.update()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    return output_path


def _call_vlm(b64_v1: str, b64_deletions: str) -> dict:
    """Send the construction drawing + deletions-only diff to Claude.

    Returns a dict with:
      - differences: list[str]   — missing dimensions / annotations
      - email: str               — plain-language email draft
      - recommendation: str      — "ACCEPT", "REVISE AND RESUBMIT", or "REJECT"
      - recommendation_reason: str — one-sentence rationale
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Export it or add it to a .env file in this directory."
        )

    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = (
        "You are a construction document review specialist. "
        "You compare Shop Drawings against Construction Drawings and produce clear, "
        "professional submittal review responses."
    )

    user_content = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64_v1},
        },
        {
            "type": "text",
            "text": (
                "**Image 1: Construction Drawing**\n"
                "Use this as reference to read dimension labels, callouts, and annotations."
            ),
        },
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64_deletions},
        },
        {
            "type": "text",
            "text": (
                "**Image 2: Deletions-Only Diff**\n"
                "RED pixels mark elements present in the Construction Drawing but missing "
                "from the Shop Drawing. Gray is unchanged context — ignore it.\n\n"
                "Based on these two images, produce a JSON object with exactly these four keys:\n\n"
                "1. \"differences\": array of strings — each red-highlighted dimension or annotation "
                "that is missing from the Shop Drawing, with its value and location. "
                "Read exact values from the Construction Drawing.\n\n"
                "2. \"email\": string — a short, professional email (3-5 sentences) to the submitter "
                "written in plain language. Summarize what is missing and what action is needed. "
                "No subject line, no salutation — body text only.\n\n"
                "3. \"recommendation\": string — exactly one of: \"ACCEPT\", \"REVISE AND RESUBMIT\", or \"REJECT\".\n\n"
                "4. \"recommendation_reason\": string — one concise sentence explaining the recommendation.\n\n"
                "Return ONLY the JSON object — no surrounding text. Example shape:\n"
                '{"differences": ["3\'-6\" horizontal dim at top-left missing"], '
                '"email": "The shop drawing is missing several dimensions...", '
                '"recommendation": "REVISE AND RESUBMIT", '
                '"recommendation_reason": "Critical dimensions are absent and must be added before approval."}'
            ),
        },
    ]

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )

    raw_text = message.content[0].text

    try:
        result = _parse_json(raw_text)
        if isinstance(result, dict):
            return {
                "differences":           result.get("differences", []),
                "email":                 result.get("email", ""),
                "recommendation":        result.get("recommendation", ""),
                "recommendation_reason": result.get("recommendation_reason", ""),
            }
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: surface raw text so the user still sees something
    return {
        "differences": [raw_text],
        "email": "",
        "recommendation": "",
        "recommendation_reason": "",
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.post("/analyze")
def analyze():
    tmp_dir = Path(tempfile.mkdtemp(prefix="bpdiff_"))
    try:
        construction_file = request.files.get("construction")
        shop_file = request.files.get("shop")

        _validate_upload(construction_file, "Construction Drawing")
        _validate_upload(shop_file, "Shop Drawing")

        v1_path = tmp_dir / "construction.pdf"
        v2_path = tmp_dir / "shop.pdf"
        diff_path = tmp_dir / "diff.png"

        construction_file.save(str(v1_path))
        shop_file.save(str(v2_path))

        # Per-file size guard (after save — avoids reading stream into memory)
        if v1_path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Construction Drawing exceeds the 20 MB limit")
        if v2_path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Shop Drawing exceeds the 20 MB limit")

        # --- Pixel diff pipeline ---
        app.logger.info("Running pixel diff pipeline…")
        generate_blueprint_diff(
            v1_path, v2_path, diff_path,
            dpi=100,
            blur_ksize=3,
            diff_threshold=30,
            context_alpha=0.30,
        )

        # --- Encode images for VLM + response ---
        app.logger.info("Encoding images…")
        img_v1 = pdf_to_image(v1_path, dpi=100)

        diff_bgr = cv2.imread(str(diff_path))
        if diff_bgr is None:
            raise RuntimeError("Diff image could not be read after generation")

        deletions_only = _make_deletions_only(diff_bgr)
        b64_diff = _img_to_b64_png(diff_bgr)
        del diff_bgr  # free memory — no longer needed

        b64_v1        = _img_to_b64_png(img_v1)
        b64_deletions = _img_to_b64_png(deletions_only)

        # --- Marked-up construction drawing ---
        app.logger.info("Generating marked-up construction drawing…")
        marked_v1 = _draw_highlights_on_image(img_v1, deletions_only)
        del img_v1, deletions_only  # free memory — no longer needed
        b64_marked_construction = _img_to_b64_png(marked_v1)
        del marked_v1  # free memory

        # --- VLM analysis ---
        app.logger.info("Calling Claude Vision…")
        analysis = _call_vlm(b64_v1, b64_deletions)
        del b64_v1, b64_deletions  # free memory after VLM call

        return jsonify({
            "diff_image_base64":           b64_diff,
            "differences":                 analysis["differences"],
            "email":                       analysis["email"],
            "recommendation":              analysis["recommendation"],
            "recommendation_reason":       analysis["recommendation_reason"],
            "marked_construction_base64":  b64_marked_construction,
        })

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Unexpected error during analysis")
        return jsonify({"error": f"Analysis failed: {exc}"}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "Upload too large. Maximum total size is 50 MB."}), 413


if __name__ == "__main__":
    app.run(debug=True)
