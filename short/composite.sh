#!/usr/bin/env bash
set -euo pipefail

# 1) Order your seven normalized clips here (left→right is the play order)
files=(
  "imran_khan_cover_drive_norm.mp4"
  "imran_khan_down_the_ground_norm.mp4"
  "imran_khan_hook_1_norm.mp4"
  "imran_khan_hook_2_norm.mp4"
  "imran_khan_inside_edge_norm.mp4"
  "imran_khan_off_drive_50_norm.mp4"
  "imran_khan_square_drive_norm.mp4"
)

D=1.0   # crossfade duration in seconds

# 2) Build ffmpeg -i inputs
args=()
for f in "${files[@]}"; do
  if [[ ! -f "$f" ]]; then echo "Missing file: $f" >&2; exit 1; fi
  args+=(-i "$f")
done

# 3) Get precise durations (seconds with millis)
mapfile -t durs < <(
  for f in "${files[@]}"; do
    ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$f"
  done
)

# 4) Compute cumulative offsets for each transition:
#    offset_k = (sum of durations of clips 0..k) - D
offsets=()
sum=0
for i in "${!durs[@]}"; do
  # Use awk for reliable float math
  sum=$(awk -v a="$sum" -v b="${durs[$i]}" 'BEGIN{printf "%.3f", a+b}')
  if (( i < ${#durs[@]} - 1 )); then
    off=$(awk -v s="$sum" -v d="$D" 'BEGIN{printf "%.3f", s-d}')
    offsets+=("$off")
  fi
done

# 5) Build filter_complex graph for xfade + acrossfade
vprev="[0:v]"; aprev="[0:a]"
graph=""
for i in $(seq 1 $((${#files[@]}-1))); do
  vi="[$i:v]"; ai="[$i:a]"
  vout="[v$i]"; aout="[a$i]"
  off="${offsets[$((i-1))]}"
  graph+="$vprev$vi xfade=transition=fade:duration=$D:offset=$off $vout; "
  graph+="$aprev$ai acrossfade=d=$D $aout; "
  vprev="$vout"; aprev="$aout"
done

# 6) Run ffmpeg
ffmpeg \
  "${args[@]}" \
  -filter_complex "$graph" \
  -map "$vprev" -map "$aprev" \
  -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p \
  -c:a aac -b:a 192k -movflags +faststart \
  imran_composite.mp4
