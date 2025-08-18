#!/usr/bin/env bash
set -euo pipefail

# --- config ---
FPS=25
W=1280
H=720
FADE=0.25         # seconds per fade in/out
FONT="/System/Library/Fonts/Supplemental/Arial.ttf"  # change if needed
OUT="IK_money_shot_v0.mp4"

# input order => pretty title
declare -a CLIPS=(
  "imran_khan_cover_drive_skel.mp4=>Cover Drive"
  "imran_khan_down_the_ground_skel.mp4=>Down The Ground"
  "imran_khan_hook_1_skel.mp4=>Hook (1)"
  "imran_khan_hook_2_skel.mp4=>Hook (2)"
  "imran_khan_inside_edge_skel.mp4=>Inside Edge"
  "imran_khan_off_drive_50_skel.mp4=>Off Drive (50)"
  "imran_khan_square_drive_skel.mp4=>Square Drive"
)

mkdir -p _tmp
> _tmp/list.txt

i=0
for pair in "${CLIPS[@]}"; do
  in="${pair%%=>*}"
  title="${pair##*=>}"

  [[ -f "$in" ]] || { echo "Missing: $in"; exit 1; }

  # duration (float seconds)
  dur=$(ffprobe -v error -select_streams v:0 -show_entries stream=duration \
        -of default=nw=1:nk=1 "$in" 2>/dev/null | head -n1)
  [[ -z "${dur}" || "${dur}" == "N/A" ]] && \
    dur=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$in" | head -n1)
  [[ -z "${dur}" || "${dur}" == "N/A" ]] && dur=3

  # fade-out start (avoid negative on short clips)
  st_out=$(python3 - <<PY
d=float("$dur"); f=float("$FADE")
print(max(0.10, d - f))
PY
)

  ffmpeg -hide_banner -y -i "$in" -vf "
    scale=w=${W}:h=${H}:force_original_aspect_ratio=decrease,
    pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2:black,
    fps=${FPS},
    format=yuv420p,
    drawbox=x=0:y=0:w=iw:h=60:color=black@0.55:t=fill,
    drawtext=fontfile='${FONT}':text='Imran Khan — ${title}  |  Correlation-Path Demo v0':x=(w-text_w)/2:y=20:fontsize=28:fontcolor=white,
    fade=t=in:st=0:d=${FADE},
    fade=t=out:st=${st_out}:d=${FADE}
  " -an "_tmp/clip_${i}.mp4"

  # tiny black spacer between clips (0.15s)
  ffmpeg -hide_banner -y -f lavfi -i "color=size=${W}x${H}:rate=${FPS}:color=black" -t 0.15 "_tmp/black_${i}.mp4" >/dev/null 2>&1

  printf "file '%s'\n" "_tmp/clip_${i}.mp4"  >> _tmp/list.txt
  printf "file '%s'\n" "_tmp/black_${i}.mp4" >> _tmp/list.txt
  ((i++))
done

# concat (copy, no re-encode)
ffmpeg -hide_banner -y -f concat -safe 0 -i _tmp/list.txt -c copy "$OUT"
echo "✅ Built $OUT"
