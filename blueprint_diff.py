"""
Blueprint Pixel-Diff Tool
=========================
Compares two versions of a construction blueprint (PDF) and produces a
composite image where:
  - Unchanged areas are rendered as faded grayscale (context)
  - Deletions  (in V1 only) are highlighted in RED
  - Additions  (in V2 only) are highlighted in GREEN
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np


# ---------------------------------------------------------------------------
# PDF → NumPy rasterisation
# ---------------------------------------------------------------------------

def pdf_to_image(pdf_path: str | Path, dpi: int = 300) -> np.ndarray:
    """Rasterise the first page of *pdf_path* at the given DPI and return
    a BGR NumPy array (OpenCV convention)."""
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    zoom = dpi / 72  # fitz renders at 72 DPI by default
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, 3
    )
    doc.close()
    # fitz returns RGB; OpenCV expects BGR
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Feature-based image registration (alignment)
# ---------------------------------------------------------------------------

def align_images(
    reference: np.ndarray,
    target: np.ndarray,
    max_features: int = 20_000,
    good_match_ratio: float = 0.75,
    ransac_thresh: float = 5.0,
) -> np.ndarray:
    """Align *target* to *reference* using ORB feature matching + homography.

    Falls back to SIFT if ORB yields too few inliers.  Returns the warped
    *target* image at the same size as *reference*.
    """
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    h, w = reference.shape[:2]

    for detector_name in ("ORB", "SIFT"):
        if detector_name == "ORB":
            detector = cv2.ORB_create(nfeatures=max_features)
            norm_type = cv2.NORM_HAMMING
        else:
            detector = cv2.SIFT_create(nfeatures=max_features)
            norm_type = cv2.NORM_L2

        kp1, des1 = detector.detectAndCompute(ref_gray, None)
        kp2, des2 = detector.detectAndCompute(tgt_gray, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            continue

        matcher = cv2.BFMatcher(norm_type)
        raw_matches = matcher.knnMatch(des1, des2, k=2)

        good = []
        for m, n in raw_matches:
            if m.distance < good_match_ratio * n.distance:
                good.append(m)

        if len(good) < 10:
            continue

        pts_ref = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_tgt = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_tgt, pts_ref, cv2.RANSAC, ransac_thresh)
        if H is None:
            continue

        inliers = int(mask.sum())
        print(
            f"  [{detector_name}] {len(good)} good matches, "
            f"{inliers} RANSAC inliers"
        )

        warped = cv2.warpPerspective(target, H, (w, h), borderValue=(255, 255, 255))
        return warped

    print("  Warning: feature alignment failed — falling back to simple resize")
    return cv2.resize(target, (w, h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Diff computation
# ---------------------------------------------------------------------------

def compute_diff_masks(
    gray_v1: np.ndarray,
    gray_v2: np.ndarray,
    blur_ksize: int = 3,
    threshold: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for (deletions, additions).

    *deletions*  – pixels present in V1 but absent from V2  (darker in V1)
    *additions*  – pixels present in V2 but absent from V1  (darker in V2)

    A small Gaussian blur + threshold suppresses scanner noise and
    anti-aliasing artifacts.
    """
    diff = cv2.absdiff(gray_v1, gray_v2)
    diff = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), 0)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Morphological close to merge nearby micro-differences into solid regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Deletions: line was in V1 (dark pixel) but not V2
    # Additions: line was in V2 (dark pixel) but not V1
    # Blueprint lines are *dark* on a light background, so "present" = low value.
    deletions = (mask == 255) & (gray_v1 < gray_v2)
    additions = (mask == 255) & (gray_v2 < gray_v1)

    return deletions, additions


# ---------------------------------------------------------------------------
# Composite visualisation
# ---------------------------------------------------------------------------

def build_composite(
    gray_v1: np.ndarray,
    deletions: np.ndarray,
    additions: np.ndarray,
    context_alpha: float = 0.30,
) -> np.ndarray:
    """Build an RGB composite:
      - faded grayscale background (context)
      - RED   overlay for deletions
      - GREEN overlay for additions
    """
    # Faded context layer: blend toward white
    faded = np.full_like(gray_v1, 255, dtype=np.uint8)
    faded = cv2.addWeighted(gray_v1, context_alpha, faded, 1.0 - context_alpha, 0)

    composite = cv2.cvtColor(faded, cv2.COLOR_GRAY2BGR)

    # Deletions → bright red  (B=0, G=0, R=255)
    composite[deletions] = (0, 0, 255)

    # Additions → bright green (B=0, G=255, R=0)
    composite[additions] = (0, 255, 0)

    return composite


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_blueprint_diff(
    v1_path: str | Path,
    v2_path: str | Path,
    output_path: str | Path,
    *,
    dpi: int = 300,
    blur_ksize: int = 3,
    diff_threshold: int = 30,
    context_alpha: float = 0.30,
) -> Path:
    """End-to-end pipeline: PDF → rasterise → align → diff → composite.

    Parameters
    ----------
    v1_path, v2_path : paths to the two PDF blueprint versions.
    output_path      : where to write the resulting composite PNG.
    dpi              : render resolution (>=300 recommended).
    blur_ksize       : Gaussian kernel size for noise suppression.
    diff_threshold   : intensity threshold for the binary diff mask.
    context_alpha    : opacity of the unchanged-area grayscale context
                       (0 = invisible, 1 = full contrast).

    Returns
    -------
    Path to the written output file.
    """
    v1_path = Path(v1_path)
    v2_path = Path(v2_path)
    output_path = Path(output_path)

    print(f"[1/5] Rasterising {v1_path.name} at {dpi} DPI …")
    img_v1 = pdf_to_image(v1_path, dpi=dpi)

    print(f"[2/5] Rasterising {v2_path.name} at {dpi} DPI …")
    img_v2 = pdf_to_image(v2_path, dpi=dpi)

    # Ensure identical canvas size (use V1 as the reference dimension)
    h, w = img_v1.shape[:2]
    if img_v2.shape[:2] != (h, w):
        img_v2 = cv2.resize(img_v2, (w, h), interpolation=cv2.INTER_AREA)

    print("[3/5] Aligning V2 to V1 …")
    img_v2_aligned = align_images(img_v1, img_v2)

    print("[4/5] Computing pixel diff …")
    gray_v1 = cv2.cvtColor(img_v1, cv2.COLOR_BGR2GRAY)
    gray_v2 = cv2.cvtColor(img_v2_aligned, cv2.COLOR_BGR2GRAY)
    deletions, additions = compute_diff_masks(
        gray_v1, gray_v2, blur_ksize=blur_ksize, threshold=diff_threshold
    )

    print("[5/5] Building composite image …")
    composite = build_composite(
        gray_v1, deletions, additions, context_alpha=context_alpha
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), composite)
    print(f"Done → {output_path}  ({composite.shape[1]}×{composite.shape[0]} px)")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a colour-coded pixel diff between two blueprint PDFs."
    )
    parser.add_argument("v1", help="Path to the V1 (older) PDF blueprint")
    parser.add_argument("v2", help="Path to the V2 (newer) PDF blueprint")
    parser.add_argument(
        "-o", "--output",
        default="diff_output.png",
        help="Output image path (default: diff_output.png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Render DPI for the PDFs (default: 300)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=30,
        help="Diff threshold 0-255 (default: 30)",
    )
    parser.add_argument(
        "--context-alpha",
        type=float,
        default=0.30,
        help="Opacity of the unchanged-area context layer (default: 0.30)",
    )
    args = parser.parse_args()

    generate_blueprint_diff(
        args.v1,
        args.v2,
        args.output,
        dpi=args.dpi,
        diff_threshold=args.threshold,
        context_alpha=args.context_alpha,
    )


if __name__ == "__main__":
    main()
