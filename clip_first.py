#!/usr/bin/env python3
"""
Clip the first N seconds of every file matching *_first_<N>.mp4,
outputting a new file with the `_first_<N>` segment removed.

Examples:
  in:  my_clip_first_12.mp4  -> out: my_clip.mp4   (keeps first 12 s)

Usage:
  python clip_first.py [--dir PATH] [--overwrite] [--reencode] [--dry-run]

Notes:
  - Defaults to stream copy (-c copy) for speed/quality.
  - If stream copy fails, auto-fallback re-encodes (H.264 + AAC).
"""
import argparse
import re
import shutil
import subprocess
from pathlib import Path

PATTERN = re.compile(r'^(?P<stem>.+)_first_(?P<sec>\d+)\.mp4$', re.IGNORECASE)

def run(cmd):
    """Run a subprocess command, returning (returncode, combined_output)."""
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    return proc.returncode, proc.stdout

def clip_file(in_path: Path, seconds: int, out_path: Path,
              overwrite: bool, force_reencode: bool) -> bool:
    """
    Return True on success. Tries stream-copy first unless force_reencode, then re-encode.
    """
    # Base ffmpeg args
    write_flag = "-y" if overwrite else "-n"

    if not force_reencode:
        # Fast path: container-level trim with stream copy
        cmd_copy = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", write_flag,
            "-i", str(in_path),
            "-t", str(seconds),
            "-map", "0",
            "-c", "copy",
            str(out_path)
        ]
        code, out = run(cmd_copy)
        if code == 0:
            return True
        print(f"[warn] Stream copy failed for '{in_path.name}'. Falling back to re-encode.\n{out}")

    # Fallback / forced re-encode: H.264 + AAC
    cmd_reenc = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", write_flag,
        "-i", str(in_path),
        "-t", str(seconds),
        "-map", "0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        str(out_path)
    ]
    code, out = run(cmd_reenc)
    if code == 0:
        return True
    print(f"[error] Re-encode also failed for '{in_path.name}'.\n{out}")
    return False

def main():
    ap = argparse.ArgumentParser(description="Clip first N seconds based on filename pattern *_first_<N>.mp4")
    ap.add_argument("--dir", "-d", default=".", help="Directory to scan (default: current directory)")
    ap.add_argument("--overwrite", "-o", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--reencode", action="store_true", help="Force re-encode instead of stream copy")
    ap.add_argument("--dry-run", action="store_true", help="Print what would happen, do not run ffmpeg")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH. Please install ffmpeg and try again.")
        raise SystemExit(1)

    base_dir = Path(args.dir).expanduser().resolve()
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        raise SystemExit(1)

    candidates = sorted(p for p in base_dir.glob("*_first_*.mp4") if p.is_file())
    matched = []
    for p in candidates:
        m = PATTERN.match(p.name)
        if not m:
            continue
        stem = m.group("stem")
        seconds = int(m.group("sec"))
        if seconds <= 0:
            print(f"[skip] Non-positive seconds in '{p.name}'")
            continue
        out_name = f"{stem}.mp4"
        out_path = p.with_name(out_name)
        matched.append((p, seconds, out_path))

    if not matched:
        print("No files matching pattern *_first_<N>.mp4 found.")
        return

    print(f"Found {len(matched)} file(s):")
    for inp, secs, outp in matched:
        print(f"  - {inp.name:45s} -> {outp.name:30s} [{secs} s]")

    if args.dry_run:
        print("\nDry run complete. No changes made.")
        return

    for inp, secs, outp in matched:
        if outp.exists() and not args.overwrite:
            print(f"[skip] Output exists, use --overwrite to replace: {outp.name}")
            continue
        ok = clip_file(inp, secs, outp, overwrite=args.overwrite, force_reencode=args.reencode)
        if ok:
            print(f"[done] {outp.name}")
        else:
            print(f"[fail] {inp.name}")

if __name__ == "__main__":
    main()
