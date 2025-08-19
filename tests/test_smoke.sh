#!/usr/bin/env bash
set -euo pipefail

mkdir -p _tmp _out

# make 3 tiny test clips (480x270@30) inside _tmp
ffmpeg -y -f lavfi -t 1 -i color=c=red:s=480x270:r=30 _tmp/a.mp4  >/dev/null 2>&1
ffmpeg -y -f lavfi -t 1 -i color=c=green:s=480x270:r=30 _tmp/b.mp4 >/dev/null 2>&1
ffmpeg -y -f lavfi -t 1 -i color=c=blue:s=480x270:r=30 _tmp/c.mp4  >/dev/null 2>&1

# IMPORTANT: entries are relative to the list file location (_tmp/)
cat > _tmp/list.txt <<EOF
file 'a.mp4'
file 'b.mp4'
file 'c.mp4'
EOF

# bash version (copy or encode fallback)
./scripts/montage.sh -l _tmp/list.txt -o _out/smoke_copy.mp4

# python version with normalize + re-encode
./scripts/montage.py -l _tmp/list.txt --normalize 480x270@30 -o _out/smoke_encode.mp4

[[ -s _out/smoke_copy.mp4 ]] && echo "bash smoke OK"
[[ -s _out/smoke_encode.mp4 ]] && echo "python smoke OK"
