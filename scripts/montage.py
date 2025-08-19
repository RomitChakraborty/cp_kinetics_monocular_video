#!/usr/bin/env python3
import argparse, subprocess, shlex, sys, tempfile, os

def run(cmd):
    print(">>", cmd)
    return subprocess.run(shlex.split(cmd))

def normalize_inputs(list_file, wh_fps, label=None, crf=18, preset="veryfast"):
    W, H, FPS = wh_fps
    list_dir = os.path.dirname(os.path.abspath(list_file))
    tmpdir = tempfile.mkdtemp(prefix="montage_norm_")
    nlist = os.path.join(tmpdir, "nlist.txt")
    with open(list_file) as f, open(nlist, "w") as out:
        for i, line in enumerate(l for l in f if l.strip()):
            path = line.strip().replace("file ", "").strip("'")
            if not os.path.isabs(path):                           
                path = os.path.join(list_dir, path)   
            if not os.path.isfile(path):
                sys.exit(f"Missing: {path}")
            outp = os.path.join(tmpdir, f"n_{i}.mp4")
            if label:
                vf = (
                    f"[0:v]fps={FPS},scale={W}:{H}:flags=bicubic,setsar=1,format=yuv420p,"
                    f"drawtext=fontfile='/System/Library/Fonts/Supplemental/Arial.ttf':"
                    f"text='{label}':fontcolor=white:fontsize=22:x=12:y=h-28:"
                    f"box=1:boxcolor=black@0.55:boxborderw=6[v]"
                )
                cmd = f"ffmpeg -y -i '{path}' -filter_complex \"{vf}\" -map \"[v]\" -an -c:v libx264 -crf {crf} -preset {preset} '{outp}'"
            else:
                cmd = f"ffmpeg -y -i '{path}' -vf \"fps={FPS},scale={W}:{H}:flags=bicubic,setsar=1,format=yuv420p\" -an -c:v libx264 -crf {crf} -preset {preset} '{outp}'"
            rc = run(cmd).returncode
            if rc != 0:
                sys.exit("Normalization failed.")
            out.write(f"file '{outp}'\n")
    return nlist

def parse_whfps(s):
    try:
        wh, fps = s.split("@")
        w, h = wh.split("x")
        return int(w), int(h), int(fps)
    except Exception:
        raise argparse.ArgumentTypeError("Use WxH@FPS, e.g. 480x270@30")

def main():
    ap = argparse.ArgumentParser(description="Concatenate videos from list file.")
    ap.add_argument("-l","--list", required=True, help="ffmpeg concat demuxer list file")
    ap.add_argument("-o","--out", default="_out/concat.mp4")
    ap.add_argument("--force-encode", action="store_true")
    ap.add_argument("--normalize", type=parse_whfps, help="e.g., 480x270@30")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--xfade", type=float, help="crossfade seconds between clips")
    ap.add_argument("--label", help="draw same label on each clip")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    list_file = args.list
    # Optional normalization
    if args.normalize:
        list_file = normalize_inputs(list_file, args.normalize, label=args.label, crf=18, preset="veryfast")

    # If xfade requested, build a filter graph concat with re-encode.
    if args.xfade:
        # Build command: inputs are the normalized ones
        with open(list_file) as f:
            paths = [ln.strip().replace("file ","").strip("'") for ln in f if ln.strip()]
        # Prepare -i args
        in_args = " ".join([f"-i '{p}'" for p in paths])
        # Build filter chain
        fps = args.normalize[2] if args.normalize else 30
        chains = [f"[{i}:v]fps={fps},format=yuv420p[v{i}];" for i in range(len(paths))]
        if len(paths) == 1:
            mapout = "[v0]"
            fc = "".join(chains)
        else:
            fc = "".join(chains)
            # pairwise xfade chain
            dur = args.xfade
            offset = max(0.0, (1.0 * (args.normalize[2] if args.normalize else 30)) - dur)  # just a placeholder offset
            fc += f"[v0][v1]xfade=transition=fade:duration={dur}:offset={offset}[x1];"
            last = "x1"
            for i in range(2, len(paths)):
                # cascade, keep same duration per segment; simple layout
                fc += f"[{last}][v{i}]xfade=transition=fade:duration={dur}:offset={(offset)*i}[x{i}];"
                last = f"x{i}"
            mapout = f"[{last}]"

        cmd = (
            f"ffmpeg -y {in_args} -filter_complex \"{fc}\" "
            f"-map \"{mapout}\" -c:v libx264 -crf {args.crf} -preset {args.preset} -pix_fmt yuv420p -movflags +faststart "
            f"\"{args.out}\""
        )
        rc = run(cmd).returncode
        sys.exit(rc)

    # Otherwise: try stream copy then fallback
    if not args.force_encode and not args.normalize and not args.label:
        rc = run(f"ffmpeg -y -f concat -safe 0 -i '{list_file}' -c copy '{args.out}'").returncode
        if rc == 0:
            sys.exit(0)
        print("Stream copy failed; falling back to re-encode...")

    rc = run(
        f"ffmpeg -y -f concat -safe 0 -i '{list_file}' "
        f"-vsync cfr -pix_fmt yuv420p -c:v libx264 -crf {args.crf} -preset {args.preset} -movflags +faststart "
        f"'{args.out}'"
    ).returncode
    sys.exit(rc)

if __name__ == "__main__":
    main()
