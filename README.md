# Imran Khan — Batting: Technique & Style (LoRA + Pose)

A compact research playground to model Imran Khan’s batting style from grainy broadcast footage: we normalize and tile video exemplars, extract skeletal motion, and explore physics‑aware signals that inform fine‑tuning (LoRA) and upscaling.

---

## 🎬 Six canonical shots (2×3, synchronized)

<video src="docs/assets/ik_shots_2x3.mp4" controls playsinline muted loop style="width:100%; max-width:960px; border:1px solid #ddd; border-radius:6px;"></video>

Shots (left→right, top→bottom): cover drive, down the ground, hook 2, inside edge, off‑drive (50), square drive.

> Want independent tiles with a synchronized “Play All” button?  
> Open the [demo page](docs/index.html) in this repo.

---

## 🦴 Corresponding skeletons (2×3, synchronized)

<video src="docs/assets/ik_skeletons_2x3.mp4" controls playsinline muted loop style="width:100%; max-width:960px; border:1px solid #ddd; border-radius:6px;"></video>

Skeletons are derived from the above six clips with a consistent fps, crop, and duration.

---

## 🔬 Physics & math (working notes)

**Pose & alignment.** Let \( \mathbf{x}_j(t)\in\mathbb{R}^2 \) be the 2D keypoint for joint \( j \) at time \( t \). We Procrustes‑align sequences to remove camera pan/zoom:

\[
\min_{s,\mathbf{R},\mathbf{t}} \sum_{t,j}
\left\lVert \mathbf{x}^{(i)}_{j}(t) - \big(s\,\mathbf{R}\,\mathbf{x}^{(\mathrm{ref})}_{j}(t)+\mathbf{t}\big) \right\rVert_2^2,
\quad \mathbf{R}\in \mathrm{SO}(2).
\]

**Kinematics.** Centered differences approximate velocities/accelerations:

\[
\dot{\mathbf{x}}_j(t) \approx \frac{\mathbf{x}_j(t+\Delta t)-\mathbf{x}_j(t-\Delta t)}{2\Delta t}, \quad
\ddot{\mathbf{x}}_j(t) \approx \frac{\mathbf{x}_j(t+\Delta t)-2\mathbf{x}_j(t)+\mathbf{x}_j(t-\Delta t)}{\Delta t^2}.
\]

Optionally smooth with a Savitzky–Golay filter to reduce jitter before differentiating.

**Bat head speed & angular velocity.** If \( \mathbf{b}(t) \) is the bat tip proxy and \( \mathbf{h}(t) \) a hand/wrist proxy,

\[
\omega_z(t) = \frac{d}{dt}\,\arg\!\big(\mathbf{b}(t)-\mathbf{h}(t)\big),\qquad
v_{\text{tip}}(t)=\left\lVert \dot{\mathbf{b}}(t) \right\rVert_2.
\]

**Temporal alignment (DTW).** To compare two shots \( \{\mathbf{f}_i\} \) and \( \{\mathbf{g}_j\} \) (e.g., feature vectors of joints),

\[
D(i,j)=\lVert \mathbf{f}_i - \mathbf{g}_j \rVert_2^2 + \min\{D(i-1,j),\,D(i,j-1),\,D(i-1,j-1)\},
\]
with \( D(0,0)=0 \). The resulting path explains those `*_corrpath.png` artifacts.

**LoRA for style.** With a base image diffusion model parameter \( \mathbf{W} \) and LoRA rank \( r \),
\[
\mathbf{W}'=\mathbf{W}+\Delta\mathbf{W},\quad \Delta\mathbf{W}=\alpha\,\frac{1}{r}\,\mathbf{A}\mathbf{B},
\]
\( \mathbf{A}\in\mathbb{R}^{d\times r}, \mathbf{B}\in\mathbb{R}^{r\times k} \). We train LoRA via the standard denoising objective
\[
\mathcal{L} = \mathbb{E}_{\mathbf{x},\epsilon,t}\Big[
\big\lVert \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x} + \sqrt{1-\bar{\alpha}_t}\epsilon, t, \mathbf{c}) \big\rVert_2^2
\Big],
\]
conditioning \( \mathbf{c} \) on either text (“Imran Khan batting”) and/or **pose** (Control‑like conditioning) to bias motion/style.

---

## 🧪 Reproducibility

- **Normalization pipeline** builds the composites shown above:
  ```bash
  ./scripts/build_composites.sh
