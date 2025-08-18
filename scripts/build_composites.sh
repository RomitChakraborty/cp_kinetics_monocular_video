#!/usr/bin/env bash
# build_composites.sh — normalize six originals + six skeletons and build 2×3 composites
# Layout assumptions:
#  - Originals: prefer short/*_norm.mp4, else ./<name>.mp4
#  - Skeletons: prefer skeleton_videos/*_skel.mp4, else ./<name>_skel.mp4

set -Eeuo pipefail

# ---------- Config ----------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHORT_DIR="$ROOT_DIR/short"
SKEL_DIR="$ROOT_DIR/skeleton_videos"
NORM_ORIG_DIR="$ROOT_DIR/data/videos_norm"
NORM_SKEL_DIR="$ROOT_DIR/data/videos_skeleton_norm"
DOCS_ASSETS="$ROOT_DIR/docs/assets"

# Six canonical basenames (hook_2 only)
SHOTS=(
  "imran_khan_cover_drive"
  "imran_khan_down_the_ground"
  "imran_khan_hook_2"
  "imran_khan_inside_edge"
  "imran_khan_off_drive_50"
  "imran_khan_square_drive"
)

# Target canvas
W=480     # even
H=270     # even
FPS=30

# ---------- Helpers ----------
log() { printf "\033[1;34m[build]\033[0m %s\n" "$*"; }
die() { printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"; }

# Prefer container duration; fallback to stream duration; print seconds (float) or 0
probe_one_duration() {
  local f="$1" dur=""
  dur=$(ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "$f" 2>/dev/null || true)
  if [[ -z "$dur" || "$dur" == "N/A" ]]; then
    dur=$(ffprobe -v error -select_streams v:0 -show_entries stream=duration -of default=nk=1:nw=1 "$f" 2>/dev/null || true)
  fi
  if [[ -z "$dur" || "$dur" == "N/A" ]]; then dur=0; fi
  printf "%.2f\n" "$dur"
}

# Min across durations > 0; 0 if nothing valid
min_duration() {
  awk 'BEGIN{min=-1}
       { if ($1+0>0 && (min<0 || $1+0<min)) min=$1+0 }
       END{ if(min<0) min=0; printf("%.2f\n", min) }'
}

# Normalize to WxH letterboxed, fixed FPS; optionally trim to duration if >0
normalize_video() {
  local in="$1"; local out="$2"; local dur="${3:-0}"

  local -a targs=()
  if awk "BEGIN{exit !($dur>0)}"; then
    targs=(-t "$dur")
  fi

  ffmpeg -nostdin -y -hide_banner -loglevel error -i "$in" \
    -vf "fps=${FPS},scale=${W}:${H}:force_original_aspect_ratio=decrease,setsar=1,pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2" \
    -an "${targs[@]}" -pix_fmt yuv420p -movflags +faststart "$out"
}

# Build 2×3 grid with xstack — label the FINAL output as [vout], then map it
make_grid() {
  local out="$1"; shift
  local inputs=("$@") ; local n="${#inputs[@]}"
  (( n == 6 )) || die "xstack expects 6 inputs; got ${n}"

  local -a args=()
  for v in "${inputs[@]}"; do args+=(-i "$v"); done

  # Build [0:v][1:v]...[N-1:v]
  local labels=""
  for ((i=0;i<n;i++)); do labels+="[${i}:v]"; done

  # 2×3 layout: columns at x = 0, w0, 2w0 ; rows at y = 0, h0
  local layout="0_0|w0_0|2w0_0|0_h0|w0_h0|2w0_h0"

  ffmpeg -nostdin -y -hide_banner -loglevel warning \
    "${args[@]}" \
    -filter_complex "${labels}xstack=inputs=${n}:layout=${layout},format=yuv420p[vout]" \
    -map "[vout]" -movflags +faststart "$out"
}

pick_original() {
  local base="$1" cand=""
  cand="$SHORT_DIR/${base}_norm.mp4"
  if [[ -f "$cand" ]]; then echo "$cand"; return 0; fi
  cand="$ROOT_DIR/${base}.mp4"
  if [[ -f "$cand" ]]; then echo "$cand"; return 0; fi
  return 1
}

pick_skeleton() {
  local base="$1" cand=""
  cand="$SKEL_DIR/${base}_skel.mp4"
  if [[ -f "$cand" ]]; then echo "$cand"; return 0; fi
  cand="$ROOT_DIR/${base}_skel.mp4"
  if [[ -f "$cand" ]]; then echo "$cand"; return 0; fi
  return 1
}

# ---------- Main ----------
need ffmpeg
need ffprobe

mkdir -p "$NORM_ORIG_DIR" "$NORM_SKEL_DIR" "$DOCS_ASSETS"

log "Selecting inputs…"
ORIG_FILES=()
SKEL_FILES=()
for s in "${SHOTS[@]}"; do
  o=$(pick_original "$s") || die "Missing original for '$s' (looked in short/ and repo root)"
  k=$(pick_skeleton "$s") || die "Missing skeleton for '$s' (looked in skeleton_videos/ and repo root)"
  ORIG_FILES+=("$o")
  SKEL_FILES+=("$k")
done

log "Probing durations (originals)…"
DURS_ORIG=()
for f in "${ORIG_FILES[@]}"; do DURS_ORIG+=("$(probe_one_duration "$f")"); done
MIN_DUR_ORIG=$(printf "%s\n" "${DURS_ORIG[@]}" | min_duration)
(( $(awk "BEGIN{print ($MIN_DUR_ORIG>0)?1:0}") )) || die "Could not determine a positive min duration for originals."
log "→ originals min duration: ${MIN_DUR_ORIG}s"

log "Probing durations (skeletons)…"
DURS_SKEL=()
for f in "${SKEL_FILES[@]}"; do DURS_SKEL+=("$(probe_one_duration "$f")"); done
MIN_DUR_SKEL=$(printf "%s\n" "${DURS_SKEL[@]}" | min_duration)
(( $(awk "BEGIN{print ($MIN_DUR_SKEL>0)?1:0}") )) || die "Could not determine a positive min duration for skeletons."
log "→ skeletons min duration: ${MIN_DUR_SKEL}s"

# If you want both composites to have identical length, uncomment:
# MIN_DUR_ALL=$(printf "%s\n" "${DURS_ORIG[@]}" "${DURS_SKEL[@]}" | min_duration)
# MIN_DUR_ORIG="$MIN_DUR_ALL"
# MIN_DUR_SKEL="$MIN_DUR_ALL"

log "Normalizing originals → ${NORM_ORIG_DIR}"
NORM_ORIG_OUT=()
for f in "${ORIG_FILES[@]}"; do
  base="$(basename "${f%.*}")"
  out="$NORM_ORIG_DIR/${base}_norm_${W}x${H}_${FPS}fps.mp4"
  normalize_video "$f" "$out" "$MIN_DUR_ORIG"
  NORM_ORIG_OUT+=("$out")
done

log "Normalizing skeletons → ${NORM_SKEL_DIR}"
NORM_SKEL_OUT=()
for f in "${SKEL_FILES[@]}"; do
  base="$(basename "${f%.*}")"
  out="$NORM_SKEL_DIR/${base}_norm_${W}x${H}_${FPS}fps.mp4"
  normalize_video "$f" "$out" "$MIN_DUR_SKEL"
  NORM_SKEL_OUT+=("$out")
done

[[ ${#NORM_ORIG_OUT[@]} -eq 6 ]] || die "Expected 6 normalized originals, got ${#NORM_ORIG_OUT[@]}"
[[ ${#NORM_SKEL_OUT[@]} -eq 6 ]] || die "Expected 6 normalized skeletons, got ${#NORM_SKEL_OUT[@]}"

log "Building 2×3 composite (originals)…"
OUT_ORIG="$DOCS_ASSETS/ik_shots_2x3.mp4"
make_grid "$OUT_ORIG" "${NORM_ORIG_OUT[@]}"

log "Building 2×3 composite (skeletons)…"
OUT_SKEL="$DOCS_ASSETS/ik_skeletons_2x3.mp4"
make_grid "$OUT_SKEL" "${NORM_SKEL_OUT[@]}"

log "Done."
printf "\nComposites:\n  - %s\n  - %s\n" "$OUT_ORIG" "$OUT_SKEL"
