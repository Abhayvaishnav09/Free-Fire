#!/usr/bin/env python3
"""
grey_scale test runner — each run creates a new session folder.

Put test images in:  gray_scale/images/
Results saved to:    gray_scale/output/session_YYYYMMDD_HHMMSS/<method>/

Usage:
    python script.py
    python script.py --method adaptive
    python script.py --session my_test_v2
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2

from grey_scale import (
    GRAYSCALE_METHODS,
    IMAGE_EXTENSIONS,
    compute_target_size,
    get_reference_size,
    resize_background_only,
)

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "images"
OUTPUT_ROOT = ROOT / "output"
DEFAULT_REFERENCE = INPUT_DIR / "VIND.KINGSTN_KNOCK_byGODL.MARCOsski11.png"


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def create_session_dir(output_root: Path, session_name: str | None) -> Path:
    if session_name:
        folder_name = f"session_{session_name}"
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"session_{stamp}"

    session_dir = output_root / folder_name
    suffix = 1
    while session_dir.exists():
        session_dir = output_root / f"{folder_name}_{suffix:02d}"
        suffix += 1

    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def write_session_info(
    session_dir: Path,
    *,
    input_dir: Path,
    images: list[Path],
    methods: list[str],
    target_size: tuple[int, int],
    reference: Path | None,
) -> None:
    info_path = session_dir / "session_info.txt"
    lines = [
        f"created: {datetime.now().isoformat(timespec='seconds')}",
        f"input_dir: {input_dir.resolve()}",
        f"image_count: {len(images)}",
        f"methods: {', '.join(methods)}",
        f"reference: {reference.resolve() if reference else 'built-in 581x50'}",
        f"target_size: {target_size[0]}x{target_size[1]} (background scaled, text original)",
        "images:",
    ]
    lines.extend(f"  - {p.name}" for p in images)
    info_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_method(
    method_name: str,
    images: list[Path],
    session_dir: Path,
    ref_size: tuple[int, int],
    *,
    scale_bg: bool,
    flat_output: bool = False,
) -> int:
    fn = GRAYSCALE_METHODS.get(method_name)
    if fn is None:
        print(f"Unknown method: {method_name}", file=sys.stderr)
        return 0

    out_dir = session_dir if flat_output else session_dir / method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = 0

    print(f"\n=== {method_name} ===")
    print(f"Output: {out_dir}")

    for idx, image_path in enumerate(images, start=1):
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  [{idx}] skip (unreadable): {image_path.name}")
            continue

        result, protected_mask, _bg_mask, meta = fn(bgr)

        if scale_bg:
            target = compute_target_size(
                bgr.shape[1],
                bgr.shape[0],
                ref_w=ref_size[0],
                ref_h=ref_size[1],
            )
            result = resize_background_only(result, protected_mask, target)

        out_path = out_dir / f"{image_path.stem}_{method_name}.png"
        cv2.imwrite(str(out_path), result, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        total = bgr.shape[0] * bgr.shape[1]
        fg_pct = 100.0 * int(meta.get("protected_px", 0)) / total
        chroma_mean = meta.get("chroma_mean", 0.0)
        chroma_p90 = meta.get("chroma_p90", 0.0)
        out_size = f"{result.shape[1]}x{result.shape[0]}"

        print(
            f"  [{idx}] {image_path.name} -> {out_path.name} "
            f"| out {out_size} | fg {fg_pct:.1f}% | chroma {chroma_mean:.1f}/{chroma_p90:.1f}"
        )
        ok += 1

    print(f"  Done: {ok}/{len(images)}")
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test killfeed grayscale methods.")
    parser.add_argument(
        "--method",
        nargs="+",
        choices=sorted(GRAYSCALE_METHODS.keys()),
        default=sorted(GRAYSCALE_METHODS.keys()),
        help="Method(s) to run (default: all)",
    )
    parser.add_argument(
        "--session",
        metavar="NAME",
        help="Optional session name (default: auto timestamp)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help=f"Input folder (default: {INPUT_DIR.name}/)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Output root (default: {OUTPUT_ROOT.name}/)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Reference killfeed image for target ratio",
    )
    parser.add_argument(
        "--no-scale-bg",
        action="store_true",
        help="Disable background-only resize to reference ratio",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write directly to output/<method>/ (no session folder)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    input_dir.mkdir(parents=True, exist_ok=True)

    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    if not images:
        print(f"No images in: {input_dir.resolve()}")
        print("Drop killfeed PNG/JPG into gray_scale/images/ then run again.")
        return 1

    reference = args.reference
    if reference and not reference.is_absolute():
        reference = ROOT / reference
    ref_size = get_reference_size(reference if reference and reference.is_file() else None)
    scale_bg = not args.no_scale_bg

    if args.flat_output:
        if len(args.method) != 1:
            print("Error: --flat-output requires exactly one --method", file=sys.stderr)
            return 1
        session_dir = output_root / args.method[0]
        session_dir.mkdir(parents=True, exist_ok=True)
        print(f"Flat output: {session_dir.resolve()}")
    else:
        session_dir = create_session_dir(output_root, args.session)
        write_session_info(
            session_dir,
            input_dir=input_dir,
            images=images,
            methods=args.method,
            target_size=ref_size,
            reference=reference if reference and reference.is_file() else None,
        )

    print(f"Input:   {input_dir.resolve()}")
    print(f"Images:  {len(images)}")
    print(f"Methods: {', '.join(args.method)}")
    print(f"Reference ratio: {ref_size[0]}x{ref_size[1]}")
    print(f"Scale background only: {scale_bg}")
    if not args.flat_output:
        print(f"Session: {session_dir.resolve()}")

    total_ok = 0
    for method_name in args.method:
        total_ok += run_method(
            method_name,
            images,
            session_dir,
            ref_size,
            scale_bg=scale_bg,
            flat_output=args.flat_output,
        )

    print(f"\nDone -> {session_dir.resolve()}")
    return 0 if total_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
