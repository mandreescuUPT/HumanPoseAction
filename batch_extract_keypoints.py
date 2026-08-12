"""
Batch Keypoint Extraction
=========================
Scans a directory for video files and extracts body keypoints
using selectable pose estimation backends.

Supported backends: mediapipe (default), yolo, movenet, movenet_thunder, movenet_lightning

Each video produces one JSON file in the output directory:
    <video_stem>_keypoints_full_body.json

Usage:
    python batch_extract_keypoints.py
    python batch_extract_keypoints.py --backend yolo
    python batch_extract_keypoints.py --backend movenet_lightning
    python batch_extract_keypoints.py --input ./data --output ./data/keypoints
    python batch_extract_keypoints.py --backend yolo --input ./my_videos --output ./my_keypoints --confidence 0.6
    python batch_extract_keypoints.py --extensions .mp4 .mov         (custom extensions)
    python batch_extract_keypoints.py --max-size 0                   (no resize)
"""

import argparse
import sys
import time
import os
import certifi
from pathlib import Path

# Use certifi certificates to fix SSL verification on macOS
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from pose_detection import run as run_pose_detection
from detector import AVAILABLE_BACKENDS

# ── Defaults ────────────────────────────────────────────────────────────────────
DEFAULT_INPUT  = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "keypoints"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}

# ── Helpers ─────────────────────────────────────────────────────────────────────
def discover_videos(directory: Path, extensions: set[str]) -> list[Path]:
    """Return sorted list of video files in *directory* (non-recursive)."""
    videos = [
        f for f in sorted(directory.iterdir())
        if f.is_file() and f.suffix.lower() in extensions
    ]
    return videos


def already_processed(video_path: Path, output_dir: Path) -> bool:
    """Check if a keypoints JSON already exists for this video."""
    expected = output_dir / f"{video_path.stem}_keypoints_full_body.json"
    return expected.exists()


# ── Main ────────────────────────────────────────────────────────────────────────
def batch_extract(
    input_dir: Path,
    output_dir: Path,
    mode: str,
    confidence: float,
    max_size: int,
    save_every: int,
    extensions: set[str],
    skip_existing: bool,
    backend: str,
):
    """Process every video in *input_dir* and save keypoints to *output_dir*."""

    input_dir  = input_dir.resolve()
    output_dir = output_dir.resolve()

    if not input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(input_dir, extensions)
    if not videos:
        print(f"No video files found in {input_dir}")
        print(f"  Looked for extensions: {', '.join(sorted(extensions))}")
        sys.exit(0)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Batch Keypoint Extraction")
    print(f"  Input dir  : {input_dir}")
    print(f"  Output dir : {output_dir}")
    print(f"  Backend    : {backend}")
    print(f"  Mode       : {mode}")
    print(f"  Confidence : {confidence}")
    print(f"  Max size   : {max_size if max_size > 0 else 'disabled'}")
    print(f"  Videos     : {len(videos)}")
    print(f"  Skip existing: {skip_existing}")
    print(f"{'='*60}\n")

    results = []
    skipped = 0
    failed  = []

    for idx, video_path in enumerate(videos, start=1):
        tag = f"[{idx}/{len(videos)}]"

        # Skip already-processed videos
        if skip_existing and already_processed(video_path, output_dir):
            print(f"  {tag} SKIP (exists): {video_path.name}")
            skipped += 1
            continue

        print(f"\n  {tag} Processing: {video_path.name}")
        print(f"  {'─'*50}")

        t0 = time.time()
        try:
            json_path = run_pose_detection(
                input_source=str(video_path),
                mode=mode,
                output_dir=output_dir,
                show_display=False,       # headless — no OpenCV window
                save_every=save_every,
                confidence=confidence,
                max_size=max_size,
                backend=backend,
            )
            elapsed = time.time() - t0
            results.append((video_path.name, json_path, elapsed))
            print(f"  ✓ Done in {elapsed:.1f}s → {json_path}")

        except Exception as e:
            elapsed = time.time() - t0
            failed.append((video_path.name, str(e)))
            print(f"  ✗ FAILED after {elapsed:.1f}s: {e}")

    # ── Final report ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Batch complete!")
    print(f"  Processed : {len(results)}")
    print(f"  Skipped   : {skipped}")
    print(f"  Failed    : {len(failed)}")
    if results:
        total_time = sum(r[2] for r in results)
        print(f"  Total time: {total_time:.1f}s  (avg {total_time/len(results):.1f}s/video)")
    if failed:
        print(f"\n  Failed videos:")
        for name, err in failed:
            print(f"    - {name}: {err}")
    print(f"{'='*60}\n")


# ── CLI ─────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch extract body keypoints from all videos in a directory."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(DEFAULT_INPUT),
        help=f"Directory containing input videos (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Directory to save keypoint JSONs (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["body", "face", "hands"],
        default="body",
        help="Detection mode (default: body)"
    )
    parser.add_argument(
        "--backend", "-b",
        choices=AVAILABLE_BACKENDS,
        default="mediapipe",
        help=f"Pose estimation backend (default: mediapipe). "
             f"Choices: {', '.join(AVAILABLE_BACKENDS)}. "
             f"Note: face/hands modes only work with mediapipe."
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence 0.0–1.0 (default: 0.5)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=600,
        help="Max frame edge for processing; 0 = no resize (default: 600)"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Checkpoint interval in frames; 0 = disabled (default: 500)"
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=None,
        help="Video extensions to look for (default: .mp4 .avi .mov .mkv .webm .flv .wmv .m4v)"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-process videos even if output JSON already exists"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    extensions = VIDEO_EXTENSIONS
    if args.extensions:
        extensions = {ext if ext.startswith(".") else f".{ext}" for ext in args.extensions}

    batch_extract(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        mode=args.mode,
        confidence=args.confidence,
        max_size=args.max_size,
        save_every=args.save_every,
        extensions=extensions,
        skip_existing=not args.no_skip,
        backend=args.backend,
    )
