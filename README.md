# Correlated Kinetics from Monocular Video: 

This repo implements a correlation–path analysis of batting kinetics from short monocular video. From each clip we detect 2D keypoints and a bat tip, root‑center and scale‑normalize, smooth and differentiate, estimate a single in‑plane swing axis, and compute time‑lagged correlations among segments. We then extract a single directed chain—the maximum‑weight correlation path—that summarizes how motion flows from the lower body to the bat. The representation is intentionally minimal (one swing axis, one timeline, one chain), grounded in the physics of proximal‑to‑distal sequencing and designed for legibility via economy, directionality, and salience.

---

# Installations/dependancies

Note ffmpeg is required.
Optionally, Python 3.8+ for montage.py.

## Six canonical shots (2×2, synchronized)
[![Six shots composite](docs/assets/ik_shots_2x3.gif)](docs/assets/ik_shots_2x3.mp4)

## Corresponding skeletons (2×2, synchronized)
[![Six skeletons composite](docs/assets/ik_skeletons_2x3.gif)](docs/assets/ik_skeletons_2x3.mp4)


Shots (left→right, top→bottom): hook, down the ground six, square drive, off-drive (camera placed at the back).
Skeletons are derived from the above clips with a consistent fps, crop, and duration.

---

## Working Notes

**Pose & alignment.** 

Let $\mathbf{x}_j(t)\in\mathbb{R}^2$ be the 2D keypoint for joint $j$ at time $t$. We Procrustes‑align sequences to remove camera pan/zoom.

**Kinematics.** Centered differences approximate velocities/accelerations:

$$
\dot{\mathbf{x}}_j(t) \approx \frac{\mathbf{x}_j(t+\Delta t)-\mathbf{x}_j(t-\Delta t)}{2\Delta t}, \qquad
\ddot{\mathbf{x}}_j(t) \approx \frac{\mathbf{x}_j(t+\Delta t)-2\mathbf{x}_j(t)+\mathbf{x}_j(t-\Delta t)}{\Delta t^2}.
$$

Optionally smooth with a Savitzky–Golay filter before differentiating.

**Bat head speed & angular velocity.** If $\mathbf{b}(t)$ is the bat tip proxy and $\mathbf{h}(t)$ a hand/wrist proxy,

$$
\omega_z(t) = \frac{d}{dt}\,\arg\!\big(\mathbf{b}(t)-\mathbf{h}(t)\big),\qquad
v_{\text{tip}}(t)=\left\lVert \dot{\mathbf{b}}(t) \right\rVert_2.
$$

**Temporal alignment (DTW).** Compare two shots $\{\mathbf{f}_i\}$ and $\{\mathbf{g}_j\}$:

$$
D(i,j)=\lVert \mathbf{f}_i - \mathbf{g}_j \rVert_2^2 + \min\{D(i-1,j),\,D(i,j-1),\,D(i-1,j-1)\}, \quad D(0,0)=0,
$$

yielding the correlation paths (`*_corrpath.png`).

## Montage tools

This repo now includes simple Bash and Python tools to concatenate batting clips.

- `scripts/montage.sh`  
  Bash wrapper around ffmpeg. Supports:
    * linear concatenation from a list file (`-l _tmp/list.txt`)
    * optional normalization (`--normalize 480x270@30`)
    * optional labels (`--label "Cover Drive"`)
    * auto-fallback from `-c copy` to re-encode

- `scripts/montage.py`  
  Python wrapper with additional options:
    * all features from `montage.sh`
    * optional crossfade transitions (`--xfade 0.5`)

