"""
Markup Shop Drawing
===================
Finds red-highlighted regions in a pixel diff PNG (dimensions present in the
Construction Drawing but missing from the Shop Drawing) and transfers them as
orange circle annotations onto the Shop Drawing PDF.

Usage (CLI):
    python markup_shop_drawing.py diff_output.png shop.pdf -o marked_up_shop.pdf

The coordinate transform uses the ratio of PDF page dimensions (points) to
diff image dimensions (pixels), so it is DPI-agnostic and handles page-size
differences automatically.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Red cluster detection
# ---------------------------------------------------------------------------

def find_red_clusters(
    diff_bgr: np.ndarray,
    min_area_px: int = 150,
    dilate_kernel: int = 18,
    dilate_iters: int = 2,
) -> list[tuple[float, float, float]]:
    """Detect clusters of red pixels in the diff image.

    Parameters
    ----------
    diff_bgr      : BGR diff image (output of generate_blueprint_diff).
    min_area_px   : Minimum contour area in pixels; smaller clusters are
                    treated as noise and ignored.
    dilate_kernel : Size of the dilation kernel used to merge nearby red
                    pixels into solid regions before contouring.
    dilate_iters  : Number of dilation iterations.

    Returns
    -------
    List of (center_x, center_y, radius) in pixel coordinates of the diff
    image.  Radius includes a small dilation buffer so circles are legible.
    """
    hsv = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2HSV)

    # Red wraps around 0 °/180 ° in HSV — need two ranges
    mask_lo = cv2.inRange(hsv, (0,   120, 120), (10,  255, 255))
    mask_hi = cv2.inRange(hsv, (170, 120, 120), (180, 255, 255))
    red_mask = cv2.bitwise_or(mask_lo, mask_hi)

    # Dilate to merge nearby dimension strokes into one region
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel)
    )
    dilated = cv2.dilate(red_mask, kernel, iterations=dilate_iters)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    clusters: list[tuple[float, float, float]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area_px:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        clusters.append((float(cx), float(cy), float(radius)))

    log.info(f"Detected {len(clusters)} red region(s) in diff image.")
    return clusters


# ---------------------------------------------------------------------------
# PDF annotation
# ---------------------------------------------------------------------------

def markup_pdf(
    shop_pdf_path: Path,
    clusters: list[tuple[float, float, float]],
    diff_image_shape: tuple[int, int],
    output_path: Path,
    min_radius_pts: float = 14.0,
    padding_pts: float = 6.0,
    stroke_color: tuple[float, float, float] = (0.9, 0.35, 0.0),  # orange
    stroke_width: float = 2.0,
) -> Path:
    """Add circle annotations to the shop drawing PDF.

    Coordinate transform
    --------------------
    scale_x = page_width_pts  / diff_image_width_px
    scale_y = page_height_pts / diff_image_height_px

    This is DPI-agnostic: it only depends on the ratio of PDF page dimensions
    to diff image dimensions, so any DPI used during rasterisation is handled
    automatically.

    Parameters
    ----------
    shop_pdf_path    : Path to the Shop Drawing PDF to annotate.
    clusters         : Output of find_red_clusters().
    diff_image_shape : (height, width) of the diff image in pixels.
    output_path      : Where to save the annotated PDF.
    min_radius_pts   : Minimum circle radius in PDF points (ensures legibility).
    padding_pts      : Extra padding added to each circle radius.
    stroke_color     : RGB tuple in [0, 1] range for the circle border.
    stroke_width     : Circle border width in points.
    """
    doc = fitz.open(str(shop_pdf_path))
    page = doc[0]

    img_h, img_w = diff_image_shape
    scale_x = page.rect.width  / img_w
    scale_y = page.rect.height / img_h

    log.info(
        f"Page size: {page.rect.width:.1f} × {page.rect.height:.1f} pts | "
        f"Diff image: {img_w} × {img_h} px | "
        f"Scale: {scale_x:.4f} × {scale_y:.4f} pts/px"
    )

    for i, (cx_px, cy_px, r_px) in enumerate(clusters):
        cx_pts = cx_px * scale_x
        cy_pts = cy_px * scale_y
        r_pts  = max(r_px * ((scale_x + scale_y) / 2) + padding_pts, min_radius_pts)

        rect = fitz.Rect(
            cx_pts - r_pts,
            cy_pts - r_pts,
            cx_pts + r_pts,
            cy_pts + r_pts,
        )

        annot = page.add_circle_annot(rect)
        annot.set_colors(stroke=stroke_color)
        annot.set_border(width=stroke_width)
        annot.update()

        log.info(
            f"  Circle {i + 1}: centre ({cx_pts:.1f}, {cy_pts:.1f}) pts, "
            f"radius {r_pts:.1f} pts"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    log.info(f"Saved marked-up shop drawing → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Public entry point (used by both CLI and Flask app)
# ---------------------------------------------------------------------------

def generate_markup(
    diff_png_path: Path | str,
    shop_pdf_path: Path | str,
    output_path: Path | str,
) -> Path:
    """End-to-end: read diff PNG → find red clusters → annotate shop PDF.

    Returns the path to the saved annotated PDF.
    """
    diff_png_path = Path(diff_png_path)
    shop_pdf_path = Path(shop_pdf_path)
    output_path   = Path(output_path)

    diff_bgr = cv2.imread(str(diff_png_path))
    if diff_bgr is None:
        raise FileNotFoundError(f"Cannot read diff image: {diff_png_path}")

    clusters = find_red_clusters(diff_bgr)
    if not clusters:
        log.warning("No red regions found — saving unmodified shop drawing.")
        import shutil
        shutil.copy(shop_pdf_path, output_path)
        return output_path

    return markup_pdf(
        shop_pdf_path,
        clusters,
        diff_bgr.shape[:2],  # (height, width)
        output_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Circle missing dimensions on a Shop Drawing PDF based on red "
            "regions in a pixel diff image."
        )
    )
    parser.add_argument("diff_png",  help="Path to the pixel diff PNG")
    parser.add_argument("shop_pdf",  help="Path to the Shop Drawing PDF")
    parser.add_argument(
        "-o", "--output",
        default="marked_up_shop.pdf",
        help="Output PDF path (default: marked_up_shop.pdf)",
    )
    args = parser.parse_args()

    try:
        out = generate_markup(args.diff_png, args.shop_pdf, args.output)
        print(f"Done → {out}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
