#!/usr/bin/env python3
"""
Render skeleton-only (normalized) or overlay videos from analysis outputs.

Inputs:
  - NPZ files produced by analyse_shots.py when export_npz: true:
      keys: kps (T,J,2), bat (T,2), kps_norm (T,J,2), bat_norm (T,2)
  - If *_norm arrays are missing, we compute normalization on the fly.

Usage:
  python render_skeleton_video.py --npz outputs/imran_khan_cover_drive_series.npz \
                                  --out outputs/imran_khan_cover_drive_skel.mp4 \
                                  --mode normalized --fps 30 --trail 25

  # Batch:
  python render_skeleton_video.py --glob "outputs/*_series.npz" --outdir skeleton_videos \
                                  --mode normalized --fps 30 --trail 25
"""

import argparse, glob, os
import numpy as np
import cv2

# Keep the same joint order as the analysis script
JOINT_ORDER = [
    'l_ankle','r_ankle',
    'l_knee','r_knee',
    'l_hip','r_hip','pelvis',
    'l_shoulder','r_shoulder',
    'l_elbow','r_elbow',
    'l_wrist','r_wrist'
]
IDX = {n:i for i,n in enumerate(JOINT_ORDER)}

# Skeleton edges (undirected)
SKELETON_EDGES = [
    ('l_ankle','l_knee'), ('l_knee','l_hip'), ('l_hip','pelvis'),
    ('r_ankle','r_knee'), ('r_knee','r_hip'), ('r_hip','pelvis'),
    ('l_shoulder','pelvis'), ('r_shoulder','pelvis'), ('l_shoulder','r_shoulder'),
    ('l_shoulder','l_elbow'), ('l_elbow','l_wrist'),
    ('r_shoulder','r_elbow'), ('r_elbow','r_wrist'),
]

def interp1_nans(y):
    x = np.arange(len(y))
    m = np.isfinite(y)
    if m.sum() < 2:
        return y
    out = y.copy()
    out[~m] = np.interp(x[~m], x[m], y[m])
    return out

def normalize_coords(kps_xy, bat_xy):
    # pelvis-centered, scale by median shoulder width
    pelvis = kps_xy[:, IDX['pelvis']]
    centered = kps_xy - pelvis[:, None, :]
    bat_c = bat_xy - pelvis
    sw = np.linalg.norm(kps_xy[:, IDX['r_shoulder']] - kps_xy[:, IDX['l_shoulder']], axis=1)
    scale = np.nanmedian(sw[(sw>1e-3) & np.isfinite(sw)])
    if not np.isfinite(scale) or scale <= 0: scale = 1.0
    return centered/scale, bat_c/scale

def to_canvas_coords(P, canvas=(720,720), margin=0.1):
    """Map normalized coords P (T,*,2) to pixel canvas with auto scale."""
    T = P.shape[0]
    # compute dynamic scale to fit
    m = np.isfinite(P).all(axis=-1)
    ext = np.nanmax(np.abs(P[m])) if np.any(m) else 1.0
    if ext < 1e-3: ext = 1.0
    H, W = canvas
    s = (1.0 - margin) * min(H, W) / (2*ext)
    cx, cy = W//2, H//2
    # convert (x,y) with y-up to canvas y-down
    Q = P.copy()
    Q[...,0] = cx + s * P[...,0]
    Q[...,1] = cy - s * P[...,1]
    return Q.astype(np.float32)

def draw_frame(canvas, joints_xy, bat_xy, trail_pts, colors):
    # joints
    for (a,b) in SKELETON_EDGES:
        ia, ib = IDX[a], IDX[b]
        pa, pb = joints_xy[ia], joints_xy[ib]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            cv2.line(canvas, tuple(pa.astype(int)), tuple(pb.astype(int)), colors['edge'], 2, cv2.LINE_AA)
    for j, name in enumerate(JOINT_ORDER):
        p = joints_xy[j]
        if np.isfinite(p).all():
            cv2.circle(canvas, tuple(p.astype(int)), 4, colors['joint'], -1, cv2.LINE_AA)
    # bat tip + trail
    if bat_xy is not None and np.isfinite(bat_xy).all():
        cv2.circle(canvas, tuple(bat_xy.astype(int)), 5, colors['bat'], -1, cv2.LINE_AA)
    for k in range(1, len(trail_pts)):
        p0, p1 = trail_pts[k-1], trail_pts[k]
        if p0 is None or p1 is None: continue
        cv2.line(canvas, tuple(p0.astype(int)), tuple(p1.astype(int)), colors['trail'], 2, cv2.LINE_AA)

def render_from_npz(npz_path, out_path, mode='normalized', fps=30, trail=25, bg='white'):
    data = np.load(npz_path, allow_pickle=True)
    if 'kps_norm' in data and 'bat_norm' in data:
        kps_n = data['kps_norm']
        bat_n = data['bat_norm']
    else:
        kps = data['kps']; bat = data['bat']
        # fill NaNs for stability
        for j in range(kps.shape[1]):
            kps[:, j, 0] = interp1_nans(kps[:, j, 0])
            kps[:, j, 1] = interp1_nans(kps[:, j, 1])
        bat[:, 0] = interp1_nans(bat[:, 0]); bat[:, 1] = interp1_nans(bat[:, 1])
        kps_n, bat_n = normalize_coords(kps, bat)

    T = kps_n.shape[0]
    H, W = 720, 720
    color_sets = {
        'white': {'bg':(255,255,255), 'edge':(40,40,40), 'joint':(0,0,0), 'bat':(230,80,80), 'trail':(180,180,180)},
        'black': {'bg':(0,0,0), 'edge':(210,210,210), 'joint':(255,255,255), 'bat':(60,180,255), 'trail':(120,120,120)}
    }
    colors = color_sets['black' if bg=='black' else 'white']

    # map to canvas
    kps_pix = to_canvas_coords(kps_n, canvas=(H,W))
    bat_pix = to_canvas_coords(bat_n.reshape(T,1,2), canvas=(H,W)).reshape(T,2)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not vw.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {out_path}")

    trail_buf = []
    for t in range(T):
        frame = np.full((H, W, 3), colors['bg'], dtype=np.uint8)
        b = bat_pix[t] if np.isfinite(bat_pix[t]).all() else None
        trail_buf.append(b)
        if len(trail_buf) > trail: trail_buf.pop(0)
        draw_frame(frame, kps_pix[t], b, [p for p in trail_buf if p is not None], colors)
        vw.write(frame)
    vw.release()
    print(f"[done] {out_path}")

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--npz", help="path to one *_series.npz")
    ap.add_argument("--glob", help="glob for multiple npz files (e.g., 'outputs/*_series.npz')")
    ap.add_argument("--out", help="output mp4 path (single)")
    ap.add_argument("--outdir", default="skeleton_videos", help="output directory (batch)")
    ap.add_argument("--mode", choices=["normalized"], default="normalized",
                    help="rendering mode; normalized = pelvis-centered space used in analysis")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--trail", type=int, default=25, help="bat-tip trail length in frames")
    ap.add_argument("--bg", choices=["white","black"], default="black")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.npz:
        out = args.out or os.path.join(args.outdir, os.path.basename(args.npz).replace("_series.npz", "_skel.mp4"))
        render_from_npz(args.npz, out, mode=args.mode, fps=args.fps, trail=args.trail, bg=args.bg)
    elif args.glob:
        for npz in sorted(glob.glob(args.glob)):
            out = os.path.join(args.outdir, os.path.basename(npz).replace("_series.npz", "_skel.mp4"))
            render_from_npz(npz, out, mode=args.mode, fps=args.fps, trail=args.trail, bg=args.bg)
    else:
        raise SystemExit("Provide --npz or --glob")

if __name__ == "__main__":
    main()
