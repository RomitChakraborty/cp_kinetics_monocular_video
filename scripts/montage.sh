#!/usr/bin/env bash
set -euo pipefail

# Linear concatenation from a list file.
# Usage:
#   ./scripts/montage.sh -l _tmp/list.txt -o _out/concat.mp4
# Options:
#   -l, --list FILE      concat list file (ffmpeg concat demuxer format)
#   -o, --out  FILE      output mp4 path
#   --force-encode       skip -c copy and re-encode
#   --normalize  WxH@FPS normalize each clip before concat (e.g., 480x270@30)
#   --crf N              CRF for encode (default 18)
#   --preset P           x264 preset (default veryfast)
#   --label "TEXT"       overlay the same label on every clip (tiny, bottom-left)

LIST=""
OUT="_out/concat.mp4"
FORCE_ENCODE=0
NORM=""
CRF=18
PRESET="veryfast"
LABEL=""

FONT_DEFAULT="/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_PATH="${FONT_PATH:-$FONT_DEFAULT}"

usage(){ grep -E '^#' "$0" | sed -E 's/^# ?//'; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -l|--list) LIST="$2"; shift 2;;
    -o|--out) OUT="$2"; shift 2;;
    --force-encode) FORCE_ENCODE=1; shift;;
    --normalize) NORM="$2"; shift 2;;
    --crf) CRF="$2"; shift 2;;
    --preset) PRESET="$2"; shift 2;;
    --label) LABEL="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "Unknown arg: $1"; usage;;
  esac
done

[[ -f "$LIST" ]] || { echo "List file not found: $LIST" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

# Optionally normalize each file to ensure identical params.
if [[ -n "$NORM" ]]; then
  # Parse WxH@FPS
  if [[ "$NORM" =~ ^([0-9]+)x([0-9]+)@([0-9]+)$ ]]; then
    W="${BASH_REMATCH[1]}"; H="${BASH_REMATCH[2]}"; FPS="${BASH_REMATCH[3]}"
  else
    echo "Bad --normalize value. Use WxH@FPS (e.g., 480x270@30)"; exit 1
  fi
  tmpdir="$(mktemp -d)"
  nlist="$tmpdir/nlist.txt"
  i=0
  while read -r line; do
    [[ -z "$line" ]] && continue
    f=${line#file }
    f=${f//\'/}
    [[ -f "$f" ]] || { echo "Missing: $f" >&2; exit 1; }
    out="$tmpdir/n_$i.mp4"
    if [[ -n "$LABEL" ]]; then
      ffmpeg -y -i "$f" -filter_complex \
        "[0:v]fps=${FPS},scale=${W}:${H}:flags=bicubic,setsar=1,format=yuv420p,\
         drawtext=fontfile='${FONT_PATH}':text='${LABEL}':fontcolor=white:fontsize=22:\
         x=12:y=h-28:box=1:boxcolor=black@0.55:boxborderw=6[v]" \
        -map "[v]" -an -c:v libx264 -crf 18 -preset veryfast "$out"
    else
      ffmpeg -y -i "$f" -vf "fps=${FPS},scale=${W}:${H}:flags=bicubic,setsar=1,format=yuv420p" \
        -an -c:v libx264 -crf 18 -preset veryfast "$out"
    fi
    echo "file '$out'" >> "$nlist"
    i=$((i+1))
  done < "$LIST"
  LIST="$nlist"
fi

# Try lossless concat first unless forced to encode.
if [[ $FORCE_ENCODE -eq 0 && -z "$LABEL" && -z "$NORM" ]]; then
  if ffmpeg -y -f concat -safe 0 -i "$LIST" -c copy "$OUT"; then
    echo "Wrote $OUT (stream copy)"; exit 0
  else
    echo "Stream copy failed; falling back to re-encode..."
  fi
fi

# Re-encode concat (robust).
ffmpeg -y -f concat -safe 0 -i "$LIST" \
  -vsync cfr -pix_fmt yuv420p \
  -c:v libx264 -crf "$CRF" -preset "$PRESET" -movflags +faststart \
  "$OUT"

echo "Wrote $OUT"
