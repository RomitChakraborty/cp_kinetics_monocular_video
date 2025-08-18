#!/usr/bin/env bash
set -euo pipefail

# ==== Settings ===============================================================
# Overlay mode: "screen" (default) or "key"
MODE=${MODE:-screen}
# Screen blend opacity (0..1)
OPACITY=${OPACITY:-0.90}
# If base fps can't be read, fall back to:
FPS_FALLBACK=25
# Over-pad the skeleton by this many seconds, then trim to base duration.
# (Big enough to cover any difference; trimming keeps it exact.)
TPAD_SECS=600

# Pairs: base video : skeleton video
# Add/remove lines if your set changes.
PAIRS=(
  "imran_khan_cover_drive.mp4:imran_khan_cover_drive_skel.mp4"
  "imran_khan_down_the_ground.mp4:imran_khan_down_the_ground_skel.mp4"
  "imran_khan_hook_1.mp4:imran_khan_hook_1_skel.mp4"
  "imran_khan_hook_2.mp4:imran_khan_hook_2_skel.mp4"
  "imran_khan_inside_edge.mp4:imran_khan_inside_edge_skel.mp4"
  "imran_khan_off_drive_50.mp4:imran_khan_off_drive_50_skel.mp4"
  "imran_khan_square_drive.mp4:imran_khan_square_drive_skel.mp4"
)

mkdir -p _aligned _out

# ==== Helpers ================================================================
frac_to_float() {
  # turn "30000/1001" into "29.97003", or "25/1" -> "25.00000"
  python3 - "$1" <<'PY'
import sys
x=sys.argv[1].strip()
if '/' in x:
    a,b=x.split('/')
    try:
        print(f"{float(a)/float(b):.6f}")
    except:
        print("")
else:
    try:
        print(f"{float(x):.6f}")
    except:
        print("")
PY
}

# ==== Main loop ==============================================================
for pair in "${PAIRS[@]}"; do
  base="${pair%%:*}"; skel="${pair##*:}"
  [[ -f "$base" ]] || { echo "❌ Missing base: $base"; continue; }
  [[ -f "$skel" ]] || { echo "❌ Missing skeleton: $skel"; continue; }

  name="${base%.mp4}"

  # Base fps (float) and duration (seconds float)
  r_rate=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate \
           -of default=nk=1:nw=1 "$base" 2>/dev/null || true)
  bfps=$(frac_to_float "${r_rate:-}")
  [[ -z "$bfps" ]] && bfps="$FPS_FALLBACK"

  bdur=$(ffprobe -v error -show_entries format=duration \
         -of default=nk=1:nw=1 "$base" 2>/dev/null | head -n1)
  [[ -z "$bdur" || "$bdur" == "N/A" ]] && bdur=10

  echo "▶ Aligning $skel to $base  (fps=$bfps, dur=${bdur}s)"

  # 1) Create duration-/fps-/size-matched skeleton master
  ffmpeg -hide_banner -y \
    -i "$skel" -i "$base" \
    -filter_complex "\
      [0:v][1:v]scale2ref=w=iw:h=ih:flags=bicubic[sk][base]; \
      [sk]fps=${bfps},setsar=1,format=rgba, \
          tpad=stop_mode=clone:stop_duration=${TPAD_SECS}[sk_pad]; \
      [sk_pad]trim=duration=${bdur},setpts=N/(${bfps}*TB)[sk_final] \
    " \
    -map "[sk_final]" -an -c:v libx264 -crf 18 -preset veryfast \
    "_aligned/${name}_skel_aligned.mp4"

  echo "   ✓ Wrote _aligned/${name}_skel_aligned.mp4"

  # 2) Overlay
  case "$MODE" in
    screen)
      ffmpeg -hide_banner -y \
        -i "$base" -i "_aligned/${name}_skel_aligned.mp4" \
        -filter_complex "\
          [1:v][0:v]scale2ref=w=iw:h=ih:flags=bicubic[sk][base]; \
          [base]setsar=1,format=rgb24[base_rgb]; \
          [sk]format=rgb24[sk_rgb]; \
          [base_rgb][sk_rgb]blend=all_mode=screen:all_opacity=${OPACITY},format=yuv420p[v] \
        " \
        -map "[v]" -map '0:a?' -c:v libx264 -crf 18 -preset veryfast -shortest \
        "_out/${name}_overlay.mp4"
      ;;
    key)
      ffmpeg -hide_banner -y \
        -i "$base" -i "_aligned/${name}_skel_aligned.mp4" \
        -filter_complex "\
          [1:v][0:v]scale2ref=w=iw:h=ih:flags=bicubic[sk][base]; \
          [sk]colorkey=0x000000:0.25:0.10[sk_k]; \
          [base][sk_k]overlay=0:0:format=auto,format=yuv420p[v] \
        " \
        -map "[v]" -map '0:a?' -c:v libx264 -crf 18 -preset veryfast -shortest \
        "_out/${name}_overlay.mp4"
      ;;
    *)
      echo "Unknown MODE='$MODE' (use 'screen' or 'key')"; exit 1;;
  esac

  echo "   ✅ Wrote _out/${name}_overlay.mp4"
done

echo "🎬 Done."
