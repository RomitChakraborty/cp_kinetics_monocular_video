#!/usr/bin/env python3
"""
Correlation-Path analysis for Imran Khan batting clips.

Pipeline:
  1) Video -> frames (OpenCV).
  2) Pose (2D): MediaPipe (default). Optional: MMPose (hook provided).
  3) Bat tip: classical detection (Canny+Hough) bootstrapped and tracked with LK optical flow.
     Optional: SAM-based segmentation (hook provided).
  4) Normalize coords, smooth, finite-diff velocities.
  5) Swing axis via PCA on bat-tip trace; project velocities.
  6) Time-lagged correlations, autocorr-aware significance, DAG construction.
  7) Maximum-weight correlation path; plot per shot.

Outputs:
  outputs/<video_basename>_corrpath.png
  outputs/<video_basename>_series.npz (optional exports)
"""




import os
os.environ.setdefault("MPLBACKEND", "Agg") # for non-GUI backend on macOS
import matplotlib
matplotlib.use("Agg")  # Use non-interactive Agg backend for plotting
import re
import math
import glob
import yaml
import json
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import mediapipe as mp
from scipy.signal import savgol_filter, correlate
from sklearn.decomposition import PCA
import networkx as nx
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
from skimage.morphology import skeletonize
from skimage import img_as_ubyte

# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    videos_dir: str = "."
    outputs_dir: str = "outputs"
    fps_override: Optional[float] = None      # if None, read from video
    smooth_window_sec: float = 0.20           # Savitzky-Golay window (seconds)
    smooth_poly: int = 2
    max_lag_sec: float = 0.4                  # +/- lag search window (seconds)
    corr_threshold: float = 0.40              # min correlation to consider
    alpha: float = 0.05                       # significance level
    min_keypoint_conf: float = 0.50           # drop frames with low pose conf
    bat_search_box_px: int = 220              # ROI around wrists for bat
    use_mmpose: bool = False                  # advanced: see hooks below
    use_sam: bool = False                     # advanced: see hooks below
    export_npz: bool = False                  # save arrays for later
    plot_figsize: Tuple[int, int] = (9, 5)

# ----------------------------
# Utilities
# ----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1

def central_diff(x: np.ndarray, dt: float) -> np.ndarray:
    v = np.zeros_like(x)
    v[1:-1] = (x[2:] - x[:-2]) / (2*dt)
    v[0] = v[1]
    v[-1] = v[-2]
    return v

def ar1_autocorr(x: np.ndarray) -> float:
    x = x - x.mean()
    if x.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0,1])

def fisher_z_pvalue(c: float, n_eff: float) -> float:
    # two-sided p-value via normal approx on Fisher z
    if n_eff <= 3:
        return 1.0
    z = np.arctanh(np.clip(c, -0.999999, 0.999999))
    se = 1.0 / math.sqrt(max(n_eff - 3.0, 1e-9))
    zscore = abs(z) / se
    # two-sided p from normal
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(zscore / math.sqrt(2))))
    return float(p)

# ---- NEW: simple 1D linear interpolator over NaNs ----
def interp1_nans(y: np.ndarray) -> np.ndarray:
    """Linearly interpolate finite values across NaN gaps along time axis."""
    x = np.arange(len(y))
    m = np.isfinite(y)
    if m.sum() < 2:
        return y  # not enough points to interpolate
    out = y.copy()
    out[~m] = np.interp(x[~m], x[m], y[m])
    return out

# ----------------------------
# Video IO
# ----------------------------
def read_video_frames(path: str, fps_override: Optional[float]=None) -> Tuple[List[np.ndarray], float]:
    """
    Robust video reader:
    1) Try OpenCV VideoCapture (fast path).
    2) If cv2 lacks VideoCapture or backend can't open, fall back to imageio.
    Returns frames as BGR uint8 and fps.
    """
    # --- Try OpenCV first ---
    try:
        import cv2  # ensure we have the real cv2
        if hasattr(cv2, "VideoCapture"):
            cap = cv2.VideoCapture(path)
            if cap is not None and cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps is None or fps <= 1 or np.isnan(fps):
                    fps = 30.0
                if fps_override is not None:
                    fps = fps_override
                frames = []
                ok, frame = cap.read()
                while ok:
                    frames.append(frame)  # BGR already
                    ok, frame = cap.read()
                cap.release()
                if frames:
                    return frames, fps
            # if we got here, cv2 exists but couldn’t read the file
    except Exception:
        pass  # drop to fallback

    # --- Fallback: imageio (uses ffmpeg) ---
    import imageio.v3 as iio
    try:
        meta = iio.immeta(path)
    except Exception:
        meta = {}
    fps = fps_override if fps_override is not None else float(meta.get("fps", 30.0))
    frames = []
    for frame in iio.imiter(path):  # RGB, HxWx3
        if frame.ndim == 3 and frame.shape[2] == 3:
            frames.append(frame[..., ::-1].copy())  # convert RGB->BGR
        elif frame.ndim == 2:
            # grayscale -> 3-channel BGR
            frames.append(np.stack([frame, frame, frame], axis=2))
        else:
            frames.append(np.ascontiguousarray(frame))
    if not frames:
        raise RuntimeError(f"Failed to decode video via OpenCV and imageio: {path}")
    return frames, fps


# ----------------------------
# Pose extraction (MediaPipe)
# ----------------------------
MP_IDX = {
    'nose': 0, 'l_eye': 2, 'r_eye': 5,
    'l_ear': 7, 'r_ear': 8,
    'l_shoulder': 11, 'r_shoulder': 12,
    'l_elbow': 13, 'r_elbow': 14,
    'l_wrist': 15, 'r_wrist': 16,
    'l_hip': 23, 'r_hip': 24,
    'l_knee': 25, 'r_knee': 26,
    'l_ankle': 27, 'r_ankle': 28,
    # pelvis not explicit: midpoint of hips
}

JOINT_ORDER = [
    'l_ankle', 'r_ankle',
    'l_knee', 'r_knee',
    'l_hip', 'r_hip', 'pelvis',
    'l_shoulder', 'r_shoulder',
    'l_elbow', 'r_elbow',
    'l_wrist', 'r_wrist'
]

def mediapipe_pose(frames: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      kps: (T, J, 2) in pixel coords
      conf: (T, J) confidences in [0,1]
    """
    mp_pose = mp.solutions.pose
    T = len(frames)
    J = len(JOINT_ORDER)
    kps = np.full((T, J, 2), np.nan, dtype=np.float32)
    conf = np.zeros((T, J), dtype=np.float32)

    with mp_pose.Pose(static_image_mode=False, model_complexity=2,
                      enable_segmentation=False, smooth_landmarks=True) as pose:
        for t, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if not res.pose_landmarks:
                continue
            lm = res.pose_landmarks.landmark
            # get needed landmarks
            xy = {}
            for name, idx in MP_IDX.items():
                x = lm[idx].x * frame.shape[1]
                y = lm[idx].y * frame.shape[0]
                c = lm[idx].visibility
                xy[name] = (x, y, c)
            # pelvis mid of hips
            if 'l_hip' in xy and 'r_hip' in xy:
                px = 0.5 * (xy['l_hip'][0] + xy['r_hip'][0])
                py = 0.5 * (xy['l_hip'][1] + xy['r_hip'][1])
                pc = 0.5 * (xy['l_hip'][2] + xy['r_hip'][2])
                xy['pelvis'] = (px, py, pc)

            for j, name in enumerate(JOINT_ORDER):
                if name in xy:
                    kps[t, j, 0] = xy[name][0]
                    kps[t, j, 1] = xy[name][1]
                    conf[t, j] = xy[name][2]
    return kps, conf

# ----------------------------
# Bat tip detection & tracking
# ----------------------------
def detect_bat_line(frame_gray: np.ndarray, roi: Optional[Tuple[int,int,int,int]]=None) -> Optional[Tuple[Tuple[int,int], Tuple[int,int]]]:
    """
    Detect a long, thin line (bat) via Canny + Probabilistic Hough within ROI.
    Returns endpoints ((x1,y1), (x2,y2)) in image coords, or None if not found.
    """
    img = frame_gray
    if roi is not None:
        x0,y0,w,h = roi
        img = frame_gray[y0:y0+h, x0:x0+w]
    edges = canny(img, sigma=2.0, low_threshold=20/255.0, high_threshold=60/255.0)
    lines = probabilistic_hough_line(edges, threshold=10, line_length=40, line_gap=5)
    if not lines:
        return None
    # choose the longest line as bat proxy
    best = max(lines, key=lambda ln: (ln[0][0]-ln[1][0])**2 + (ln[0][1]-ln[1][1])**2)
    (x1,y1),(x2,y2) = best
    if roi is not None:
        x1 += x0; x2 += x0; y1 += y0; y2 += y0
    return (x1,y1),(x2,y2)

def pick_bat_tip_from_line(line_pts, wrist_top_xy):
    (x1,y1),(x2,y2) = line_pts
    e1 = np.array([x1,y1], dtype=np.float32)
    e2 = np.array([x2,y2], dtype=np.float32)
    # tip = farthest endpoint from the top-hand wrist
    d1 = np.linalg.norm(e1 - wrist_top_xy)
    d2 = np.linalg.norm(e2 - wrist_top_xy)
    return e1 if d1 > d2 else e2

def choose_top_hand(wrist_L: np.ndarray, wrist_R: np.ndarray, bat_center: np.ndarray) -> np.ndarray:
    # Heuristic: "top hand" is the wrist closer to bat handle (near bat center at handle end).
    # We don't know handle end; use bat center-of-line as proxy; pick wrist closer to center.
    dL = np.linalg.norm(wrist_L - bat_center)
    dR = np.linalg.norm(wrist_R - bat_center)
    return wrist_L if dL < dR else wrist_R

def track_bat_tip(frames: List[np.ndarray], kps: np.ndarray, conf: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Returns bat_tip (T,2) in pixel coords (np.nan if unavailable).
    Strategy: detect a bat line around wrists in first usable frame; pick tip; then track with LK flow.
    If detection fails at any step, try re-detecting with updated ROI.
    """
    T = len(frames)
    H, W = frames[0].shape[:2]
    tip = np.full((T,2), np.nan, dtype=np.float32)

    # Find a frame with high-confidence wrists to bootstrap
    jL = JOINT_ORDER.index('l_wrist')
    jR = JOINT_ORDER.index('r_wrist')
    good_idxs = [t for t in range(T) if conf[t, jL] >= cfg.min_keypoint_conf and conf[t, jR] >= cfg.min_keypoint_conf]
    if not good_idxs:
        return tip

    # Try first good frame for detection
    init_t = good_idxs[0]
    gray0 = cv2.cvtColor(frames[init_t], cv2.COLOR_BGR2GRAY)

    # ROI around wrists
    wl = kps[init_t, jL]; wr = kps[init_t, jR]
    cx = int(0.5*(wl[0] + wr[0])); cy = int(0.5*(wl[1] + wr[1]))
    half = cfg.bat_search_box_px//2
    roi = (max(0, cx-half), max(0, cy-half), min(W, 2*half), min(H, 2*half))

    line = detect_bat_line(gray0, roi=roi)
    if line is None:
        # fallback: global search (rare)
        line = detect_bat_line(gray0, roi=None)
        if line is None:
            return tip

    # pick top-hand wrist and tip
    (x1,y1),(x2,y2) = line
    bat_center = np.array([(x1+x2)/2.0, (y1+y2)/2.0], dtype=np.float32)
    top_wrist = choose_top_hand(wl, wr, bat_center)
    tip0 = pick_bat_tip_from_line(line, top_wrist)
    tip[init_t] = tip0

    # Track forward/backward with LK optical flow
    lk_params = dict(winSize=(21,21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    prev_gray = gray0.copy()
    prev_pt = tip0.reshape(1,1,2).astype(np.float32)

    # forward
    for t in range(init_t+1, T):
        gray = cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY)
        new_pt, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pt, None, **lk_params)
        if st is not None and st[0,0] == 1:
            tip[t] = new_pt[0,0]
            prev_pt = new_pt
            prev_gray = gray
        else:
            # re-detect around wrists for robustness
            wl = kps[t, jL]; wr = kps[t, jR]
            if np.any(np.isnan(wl)) or np.any(np.isnan(wr)):
                prev_gray = gray
                continue
            cx = int(0.5*(wl[0] + wr[0])); cy = int(0.5*(wl[1] + wr[1]))
            roi = (max(0, cx-half), max(0, cy-half), min(W, 2*half), min(H, 2*half))
            line = detect_bat_line(gray, roi=roi)
            if line is not None:
                (x1,y1),(x2,y2) = line
                bat_center = np.array([(x1+x2)/2.0, (y1+y2)/2.0], dtype=np.float32)
                top_wrist = choose_top_hand(wl, wr, bat_center)
                re_tip = pick_bat_tip_from_line(line, top_wrist)
                tip[t] = re_tip
                prev_pt = re_tip.reshape(1,1,2).astype(np.float32)
            prev_gray = gray

    # backward
    prev_gray = gray0.copy()
    prev_pt = tip0.reshape(1,1,2).astype(np.float32)
    for t in range(init_t-1, -1, -1):
        gray = cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY)
        new_pt, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pt, None, **lk_params, flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS)
        if st is not None and st[0,0] == 1:
            tip[t] = new_pt[0,0]
            prev_pt = new_pt
            prev_gray = gray
        else:
            # attempt re-detect
            wl = kps[t, jL]; wr = kps[t, jR]
            if np.any(np.isnan(wl)) or np.any(np.isnan(wr)):
                prev_gray = gray
                continue
            cx = int(0.5*(wl[0] + wr[0])); cy = int(0.5*(wl[1] + wr[1]))
            roi = (max(0, cx-half), max(0, cy-half), min(W, 2*half), min(H, 2*half))
            line = detect_bat_line(gray, roi=roi)
            if line is not None:
                (x1,y1),(x2,y2) = line
                bat_center = np.array([(x1+x2)/2.0, (y1+y2)/2.0], dtype=np.float32)
                top_wrist = choose_top_hand(wl, wr, bat_center)
                re_tip = pick_bat_tip_from_line(line, top_wrist)
                tip[t] = re_tip
                prev_pt = re_tip.reshape(1,1,2).astype(np.float32)
            prev_gray = gray

    return tip

# ---- NEW: fallback bat tip if tracking fails, derived from pose ----
def bat_tip_fallback_from_pose(kps: np.ndarray) -> np.ndarray:
    """
    Approximate bat tip when tracking fails:
    1) Estimate swing axis from wrist motion.
    2) For each frame pick the forearm (elbow->wrist) more aligned with that axis.
    3) Tip ~ wrist + alpha * forearm (alpha ≈ 1.6).
    """
    T = len(kps)
    tip = np.full((T,2), np.nan, dtype=np.float32)
    jLW, jRW = JOINT_ORDER.index('l_wrist'), JOINT_ORDER.index('r_wrist')
    jLE, jRE = JOINT_ORDER.index('l_elbow'), JOINT_ORDER.index('r_elbow')
    jLS, jRS = JOINT_ORDER.index('l_shoulder'), JOINT_ORDER.index('r_shoulder')

    # rough swing axis from wrist paths
    wrists = []
    for j in (jLW, jRW):
        if np.isfinite(kps[:, j, :]).all():
            wrists.append(kps[:, j, :])
    if wrists:
        X = np.vstack(wrists) - np.nanmean(np.vstack(wrists), axis=0)
        u_alt = PCA(n_components=1).fit(X).components_[0]
        u_alt = u_alt / (np.linalg.norm(u_alt) + 1e-9)
    else:
        u_alt = np.array([1.0, 0.0], dtype=np.float32)

    alpha = 1.6  # heuristically ~ bat length in "forearm" units
    for t in range(T):
        lw, rw = kps[t, jLW], kps[t, jRW]
        le, re = kps[t, jLE], kps[t, jRE]
        cands = []
        if np.all(np.isfinite(lw)) and np.all(np.isfinite(le)):
            cands.append((lw, lw - le))
        if np.all(np.isfinite(rw)) and np.all(np.isfinite(re)):
            cands.append((rw, rw - re))
        if not cands:
            continue
        wrist, forearm = max(cands, key=lambda pair: abs(np.dot(pair[1], u_alt)))
        tip[t] = wrist + alpha * forearm / (np.linalg.norm(forearm) + 1e-9)
    return tip



# ----------------------------
# Core math stages
# ----------------------------
def normalize_coords(kps_xy: np.ndarray, pelvis_idx: int, l_sh_idx: int, r_sh_idx: int, bat_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    kps_xy: (T,J,2) in pixel coordinates; may contain NaNs.
    Returns normalized joint coords and bat coords (centered at pelvis; scaled by median shoulder width).
    """
    T, J, _ = kps_xy.shape
    pelvis = kps_xy[:, pelvis_idx, :]  # (T,2)
    centered = kps_xy - pelvis[:, None, :]
    bat_centered = bat_xy - pelvis

    shoulder_w = np.linalg.norm(kps_xy[:, r_sh_idx, :] - kps_xy[:, l_sh_idx, :], axis=1)
    scale = np.nanmedian(shoulder_w[np.isfinite(shoulder_w) & (shoulder_w > 1e-3)])
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    return centered/scale, bat_centered/scale, float(scale)

def swing_axis_from_bat(bat_xy: np.ndarray) -> np.ndarray:
    good = np.all(np.isfinite(bat_xy), axis=1)
    X = bat_xy[good]
    if len(X) < 5:
        # default axis if too few points: horizontal
        return np.array([1.0, 0.0], dtype=np.float32)
    pca = PCA(n_components=2).fit(X)
    u = pca.components_[0]
    return (u / np.linalg.norm(u)).astype(np.float32)

def series_from_projection(kps_n: np.ndarray, bat_n: np.ndarray, u: np.ndarray, fps: float, smooth_win_s: float, smooth_poly: int) -> Tuple[Dict[str,np.ndarray], Dict[str,int]]:
    """
    Build projected velocity series amd peak times robust to NaNs.
    Drop nodes with too few valid samples to avoid all-NaN crashes.
    Returns:
      series: dict of projected velocities per joint + 'bat'
      t_peak: dict of peak-time indices per joint + 'bat'
    """
    dt = 1.0 / fps
    T, J, _ = kps_n.shape

    # --- NEW: interpolate NaNs in positions before smoothing ---
    for j in range(J):
        for d in (0, 1):
            k = kps_n[:, j, d]
            if np.isfinite(k).sum() >= 2:
                kps_n[:, j, d] = interp1_nans(k)
    for d in (0, 1):
        b = bat_n[:, d]
        if np.isfinite(b).sum() >= 2:
            bat_n[:, d] = interp1_nans(b)

    # smoothing window in frames (odd)
    win = odd(max(5, int(round(smooth_win_s * fps))))
    def smooth2(x): 
        return savgol_filter(x, window_length=win, polyorder=smooth_poly, axis=0, mode='interp')

    # smooth positions (only on frames that are finite now)
    kps_s = np.copy(kps_n)
    bat_s = np.copy(bat_n)
    kps_s = smooth2(kps_s)
    bat_s = smooth2(bat_s)

    # velocities via central difference
    v_kps = central_diff(kps_s, dt)    # (T,J,2)
    v_bat = central_diff(bat_s, dt)    # (T,2)

    # project onto swing axis u
    vproj = np.einsum('tjd,d->tj', v_kps, u)   # (T,J)
    vbat  = np.einsum('td,d->t',  v_bat, u)    # (T,)

    # build dict: raw projected velocities
    series = {name: vproj[:, j] for j, name in enumerate(JOINT_ORDER)}
    series['bat'] = vbat

    # --- NEW: drop series with insufficient valid samples ---
    min_valid = max(10, int(0.2 * T))  # need >= 20% frames valid
    series = {k: v for k, v in series.items() if np.isfinite(v).sum() >= min_valid}

    if 'bat' not in series:
        # let caller decide whether to synthesize a fallback bat tip
        raise RuntimeError("Bat signal missing after smoothing/projection (insufficient valid samples).")

    # peak times (guarded: we filtered all-NaN series)
    t_peak = {name: int(np.nanargmax(np.abs(series[name]))) for name in series.keys()}
    return series, t_peak

def norm_xcorr(a: np.ndarray, b: np.ndarray, maxlag: int) -> Tuple[np.ndarray, np.ndarray]:
    # ensure finite
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]; b = b[mask]
    if len(a) < 5:
        lags = np.arange(-maxlag, maxlag+1)
        return lags, np.full_like(lags, np.nan, dtype=float)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    # correlate b vs a so positive lag = a leads b
    full = correlate(b, a, mode='full') / len(a)
    lags = np.arange(-len(a)+1, len(b))
    sel = (lags >= -maxlag) & (lags <= maxlag)
    return lags[sel], full[sel]

def effective_sample_size(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]; b = b[mask]
    N = len(a)
    if N < 5:
        return 1.0
    r1 = ar1_autocorr(a)
    r2 = ar1_autocorr(b)
    return float(N * (1 - r1*r2) / (1 + r1*r2 + 1e-9))

def build_correlation_dag(series: Dict[str,np.ndarray], t_peak: Dict[str,int],
                          fps: float, cfg: Config) -> nx.DiGraph:
    names = list(series.keys())
    G = nx.DiGraph()
    for n in names:
        G.add_node(n, t=int(t_peak[n]))

    maxlag = int(round(cfg.max_lag_sec * fps))

    # --- NEW: skip series with too few valid samples ---
    any_series = next(iter(series.values()))
    MIN_VALID = max(10, int(0.2 * len(any_series)))

    for i in names:
        for j in names:
            if i == j: 
                continue
            # temporal order to keep DAG
            if t_peak[i] >= t_peak[j]:
                continue
            if np.isfinite(series[i]).sum() < MIN_VALID: 
                continue
            if np.isfinite(series[j]).sum() < MIN_VALID: 
                continue

            lags, rho = norm_xcorr(series[i], series[j], maxlag=maxlag)
            if rho is None or np.all(~np.isfinite(rho)):
                continue
            k = int(np.nanargmax(rho))
            cmax = float(rho[k])
            lag = int(lags[k])
            if lag <= 0:
                continue
            # significance
            n_eff = effective_sample_size(series[i], series[j])
            p = fisher_z_pvalue(cmax, n_eff)
            if cmax >= cfg.corr_threshold and p < cfg.alpha:
                G.add_edge(i, j, weight=cmax, lag_frames=lag, pval=p)
    return G

def max_weight_path(G: nx.DiGraph, sources: List[str], sink: str) -> List[str]:
    H = G.copy()
    super_src = "__SRC__"
    H.add_node(super_src, t=min(nx.get_node_attributes(G, 't').values()))
    for s in sources:
        if s in H.nodes:
            H.add_edge(super_src, s, weight=0.0, lag_frames=0, pval=0.0)
    path = nx.algorithms.dag.dag_longest_path(H, weight='weight')
    if path and path[0] == super_src:
        path = path[1:]
    # If desired sink isn't last, trim up to sink
    if sink in path:
        idx = path.index(sink)
        path = path[:idx+1]
    return path

# ----------------------------
# Plotting
# ----------------------------
ANATOMY_ORDER = ['ankle','knee','hip','pelvis','shoulder','elbow','wrist','bat']

def anatomy_rank(name: str) -> int:
    for k, key in enumerate(ANATOMY_ORDER):
        if key in name:
            return k
    return len(ANATOMY_ORDER)

def plot_correlation_path(G: nx.DiGraph, series: Dict[str,np.ndarray], t_peak: Dict[str,int],
                          fps: float, title: str, out_path: Path, figsize=(9,5)):
    # sources = feet + pelvis if present
    sources = [n for n in G.nodes if any(tok in n for tok in ['ankle','pelvis'])]
    sink = 'bat' if 'bat' in G.nodes else max(G.nodes, key=lambda n: t_peak[n])
    path = max_weight_path(G, sources, sink)

    # node layout
    xs = {n: t_peak[n]/fps for n in G.nodes}
    ys = {n: anatomy_rank(n) for n in G.nodes}

    # figure
    plt.figure(figsize=figsize)
    # top strip: bat speed
    ax_top = plt.axes([0.10, 0.78, 0.85, 0.18])
    t = np.arange(len(series['bat']))/fps
    ax_top.plot(t, np.abs(series['bat']))
    ax_top.set_ylabel('|bat speed|')
    ax_top.set_xticks([])

    # main panel
    ax = plt.axes([0.10, 0.10, 0.85, 0.60])
    # faint all nodes
    for n in G.nodes:
        ax.scatter(xs[n], ys[n], s=40, alpha=0.35)
        ax.text(xs[n], ys[n]+0.10, n, fontsize=8, ha='center')

    # faint all edges
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 0.0)
        ax.plot([xs[u], xs[v]], [ys[u], ys[v]], linewidth=0.5 + 1.0*w, alpha=0.25)

    # highlight path
    for u, v in zip(path[:-1], path[1:]):
        data = G.get_edge_data(u, v, default={'weight':0.0, 'lag_frames':0})
        w = data.get('weight', 0.0)
        lag = data.get('lag_frames', 0)
        ax.plot([xs[u], xs[v]], [ys[u], ys[v]], linewidth=2.5 + 4.0*w, alpha=0.95)
        xm, ym = (xs[u]+xs[v])/2.0, (ys[u]+ys[v])/2.0
        ax.text(xm, ym, f"{w:.2f}, {1000*lag/fps:.0f} ms", fontsize=8, ha='center', va='center')

    ax.set_yticks(range(len(ANATOMY_ORDER)))
    ax.set_yticklabels([s.capitalize() for s in ANATOMY_ORDER])
    ax.set_xlabel('Time (s, by peak activation)')
    ax.set_title(title)
    plt.savefig(out_path, dpi=200)
    plt.close()

# ----------------------------
# Main driver per video
# ----------------------------
def analyze_video(video_path: Path, cfg: Config):
    print(f"[info] Processing: {video_path.name}")
    frames, fps = read_video_frames(str(video_path), cfg.fps_override)
    H, W = frames[0].shape[:2]
    # Pose
    kps, conf = mediapipe_pose(frames)  # default backend
    # Pelvis as index
    pelvis_idx = JOINT_ORDER.index('pelvis')
    l_sh_idx = JOINT_ORDER.index('l_shoulder')
    r_sh_idx = JOINT_ORDER.index('r_shoulder')

    # Bat tip
    bat_tip = track_bat_tip(frames, kps, conf, cfg)  # (T,2)

    
    # Filter frames with extremely low conf across core joints
    core_joints = ['l_ankle','r_ankle','l_hip','r_hip','l_shoulder','r_shoulder','l_wrist','r_wrist','pelvis']
    core_idx = [JOINT_ORDER.index(j) for j in core_joints]
    conf_core = conf[:, core_idx]
    keep = (np.nanmean(conf_core, axis=1) >= cfg.min_keypoint_conf)
    if keep.sum() < 10:
        print("[warn] Few reliable frames; results may be noisy.")
    # --- NEW: actually drop low-confidence frames from arrays ---
    kps = kps[keep]
    conf = conf[keep]
    bat_tip = bat_tip[keep]

    # --- NEW: if bat tracking is too sparse, synthesize from pose ---
    finite_frac = np.isfinite(bat_tip).all(axis=1).mean() if len(bat_tip) else 0.0
    if finite_frac < 0.40:  # <40% frames have a valid tracked tip
        print("[info] Bat tracker weak; using pose-based fallback.")
        bat_fb = bat_tip_fallback_from_pose(kps)
        use_fb = (~np.isfinite(bat_tip).all(axis=1)) & (np.isfinite(bat_fb).all(axis=1))
        bat_tip[use_fb] = bat_fb[use_fb]



    # Normalize coords
    kps_n, bat_n, scale = normalize_coords(kps, pelvis_idx, l_sh_idx, r_sh_idx, bat_tip)

    # Swing axis and series
    u = swing_axis_from_bat(bat_n)
    series, t_peak = series_from_projection(kps_n, bat_n, u, fps, cfg.smooth_window_sec, cfg.smooth_poly)

    # DAG
    G = build_correlation_dag(series, t_peak, fps, cfg)

    # --- QC metrics ---
    n_frames = len(next(iter(series.values())))
    finite_bat = int(np.isfinite(series['bat']).sum())
    path_nodes = max_weight_path(G, [n for n in G.nodes if 'ankle' in n or 'pelvis' in n], 'bat')
    path_weight = sum(G[u][v]['weight'] for u, v in zip(path_nodes[:-1], path_nodes[1:]) if G.has_edge(u,v))
    print(f"[qc] frames={n_frames}, bat_finite={finite_bat}, nodes={len(G.nodes)}, edges={len(G.edges)}, "
      f"path_len={len(path_nodes)}, path_weight={path_weight:.2f}")


    # Plot
    out_png = Path(cfg.outputs_dir) / f"{video_path.stem}_corrpath.png"
    plot_correlation_path(G, series, t_peak, fps, title=video_path.stem, out_path=out_png, figsize=cfg.plot_figsize)
    print(f"[done] Saved: {out_png}")

    # Optional export
    if cfg.export_npz:
        out_npz = Path(cfg.outputs_dir) / f"{video_path.stem}_series.npz"
        np.savez_compressed(out_npz, kps=kps, conf=conf, bat=bat_tip, kps_norm=kps_n, bat_norm=bat_n,
                            u=u, series={k: series[k] for k in series})
        print(f"[done] Saved arrays: {out_npz}")

def find_videos(dir_path: Path) -> List[Path]:
    vids = []
    for p in sorted(dir_path.glob("*.mp4")):
        if "archive" in p.parts:
            continue
        vids.append(p)
    return vids

def main():
    # load config if present
    cfg_path = Path("config.yaml")
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            data = yaml.safe_load(f)
        cfg = Config(**{**Config().__dict__, **(data or {})})
    else:
        cfg = Config()
    outdir = Path(cfg.outputs_dir)
    ensure_dir(outdir)
    videos = find_videos(Path(cfg.videos_dir))
    if not videos:
        print("[warn] No .mp4 files found in directory.")
        return
    for v in videos:
        analyze_video(v, cfg)
    print("[info] All done.")

if __name__ == "__main__":
    main()
