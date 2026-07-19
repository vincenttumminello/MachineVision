# MCHA4400 toolchain — dockerised

The Lab 1 "Toolchain setup" (clang, cmake, ninja, cppcheck, doctest, nanobench,
boost, Eigen3, autodiff, SuiteSparse, OpenCV, VTK, ONNX Runtime, doxygen)
packaged as a container. No Homebrew, no `apt`/`update-alternatives` dance, no
"restart your terminal" — one image, reproducible everywhere.

## What you need on the host

Already present on this machine:

- Docker
- An X server. On Wayland (Fedora KDE), XWayland provides this automatically via
  `DISPLAY=:0`, so GUI windows just work.

## Two ways to use it

### 1. VS Code (recommended — matches the course's "VS Code as IDE" section)

1. Open `~/MCHA4400` in VS Code.
2. Install the **Dev Containers** extension if you haven't.
3. `F1` → **Dev Containers: Reopen in Container**.

VS Code builds the image, mounts this folder at `/workspace`, and installs
clangd + CMake Tools inside. IntelliSense, build, and debug all run against the
containerised toolchain. Your files stay on the host, owned by you.

### 2. Terminal

```bash
./mcha4400 build          # build the image once (slow: pulls OpenCV/VTK/Boost)
./mcha4400                # interactive shell in the toolchain, at /workspace
./mcha4400 cmake --version
```

Typical build of a course project from the host:

```bash
./mcha4400 bash -c 'cmake -S . -B build -G Ninja && cmake --build build'
```

## Verifying the GUI works

Inside the container (`./mcha4400`):

```bash
xeyes                     # X11 forwarding — a pair of eyes should pop up
glxinfo | grep "renderer" # OpenGL — should name your GPU (or llvmpipe)
```

If OpenGL/VTK misbehaves, force software rendering:

```bash
LIBGL_ALWAYS_SOFTWARE=1 ./mcha4400 <your-command>
```

## Package sources

Most packages come from Ubuntu 24.04 `apt`. The three that aren't packaged —
`autodiff`, `nanobench`, `onnxruntime` — are installed from upstream in the
Dockerfile (pinned versions at the top as build args). That's the only reason
the lab reaches for Homebrew; here it's baked in.

- `find_package(autodiff)`, `find_package(nanobench)` work out of the box.
- ONNX Runtime headers/libs are under `/usr/local` (`onnxruntime_cxx_api.h`,
  `libonnxruntime.so`); link against `onnxruntime`.
- VTK comes from `apt`, but Ubuntu's `libvtk9-dev` ships a CMake config that
  trips `find_package(VTK)` on modern CMake. The image installs VTK's external
  dev dependencies and a `vtk-find-package-helpers.cmake` shim (see
  `.devcontainer/`) so `find_package(VTK COMPONENTS ...)` just works — no change
  needed in your project's `CMakeLists.txt`.

## Notes

- The container user is `dev` with your UID/GID (1000), so files created under
  `/workspace` are owned by you on the host.
- Maps to Lab 1 troubleshooting §5.3/§5.4 (imshow/GTK): the X socket and
  `DISPLAY` are already wired up, so those errors shouldn't occur.

---

# RoboCup field localisation (`nubots/`)

A recursive Bayesian field localiser for a RoboCup humanoid, built on the
course's square-root information Gaussian framework. It replaces the NUbots
per-frame NLopt point estimate with a proper filter that carries calibrated
uncertainty, and it beats that baseline on OptiTrack ground truth:
**0.107 m / 5.4° RMSE vs 0.506 m / 11.3°** on the `data2` mocap recording,
at ~0.5 ms per update on a laptop CPU.

## State and belief representation

The state is 8-dimensional:

| indices | meaning |
|---------|---------|
| 0–2     | torso position in the field frame `{f}` [m] |
| 3–5     | torso roll, pitch, yaw [rad] |
| 6–7     | camera-mount attitude bias (roll, pitch) [rad] |

The belief is a Gaussian in square-root information form
(`GaussianInfo.hpp`), so covariance never loses positive-definiteness and
marginals/conditionals are cheap. Every vision update is a MAP optimisation
(trust-region Newton, `funcmin.hpp`) whose Laplace approximation becomes the
posterior — i.e. an iterated-EKF-style update rather than a single
linearisation (`Measurement.cpp`).

The camera-bias states absorb the systematic ~1–3° mount/kinematics pitch
error discovered in the recorded data (field-line/landmark residuals growing
with range²). They are a slow random walk applied on the camera side of the
extrinsic, `Tfc = Tfb(x) · Tbc · R(δc)`, and are estimated jointly with the
pose (`SystemLocalisation.h`).

## Prediction: odometry velocity + gyroscope attitude rate

Prediction integrates a buffered body twist through an SDE with RK4
(`SystemEstimator.cpp`); between vision frames the belief coasts on this
twist, so catching up a backlog after a slow bootstrap is trivial.

The twist is assembled per odometry interval
(`SystemLocalisation::twistFromOdometry`):

- **Linear velocity** from finite-differencing the walk-engine odometry
  stream `Htw` — position odometry is excellent (sub-mm while standing).
- **Angular velocity from the torso gyroscope**, not from odometry. This
  matters: the walk-engine attitude *slips badly while turning* — on `data2`
  the odometry alone loses ~150° of yaw by t = 40 s (mocap-verified), while
  the gyro tracks every turn and only drifts at a smooth bias-like rate.
  The gyro bias is self-calibrated from "quiet" samples (|ω| < 0.05 rad/s,
  i.e. the robot standing still reads its own bias); the odometry-differenced
  rate remains as a fallback when the gyro is not finite.

Process noise is a diagonal PSD (`Parameters` in `SystemLocalisation.h`),
tuned so the filter stays *consistent*: the mocap truth lies inside the
3-sigma position bound on 100% of `data2` samples.

## Initialisation: first-frame grid solve + start-half prior

No baseline pose is used. `solveInitialPose()` (`fieldLocalisation.cpp`)
grid-searches (x, y, yaw) on the first usable vision frame (z/roll/pitch come
from kinematics), scoring each cell with the robust landmark log-likelihood
(~170 ms once). On-field landmarks are invariant under the field's 180°
rotation, so the global maximum has an equally-likely mirror; the
GameController-style start-in-own-half prior (`ownHalfXSign`) picks the side.
The filter then starts from a deliberately loose prior and sharpens
recursively.

## Measurement models

Applied per vision frame through `system.process()` (which also drives the
optional Gaussian-mixture hypothesis bank):

- **Field landmarks** (`MeasurementFieldLandmarks`): YOLO detections arrive
  as calibrated unit rays in the camera frame (the fisheye model is already
  applied upstream), classes = goal posts and L/T/X line intersections.
  Rays are gated (20°) and nearest-neighbour associated against the known
  field map at the prior mean; the likelihood is a robust inlier-Gaussian +
  uniform-clutter mixture on the unit sphere (σ = 0.25 rad, inlier weight
  0.7), so gross mis-detections are absorbed rather than fought.
- **Gravity** (`MeasurementGravity`): accelerometer direction observes roll
  and pitch.
- **Kinematic height** (`MeasurementKinematicHeight`): torso height from the
  leg kinematic chain observes z.
- **Field lines** exist (`MeasurementFieldLines`) but ship disabled: their
  per-frame errors are strongly correlated and multi-line capture degraded
  accuracy on the recorded data.

## Side disambiguation (`SideDisambiguator`, `OutOfFieldFeatures`)

The field is symmetric under 180° rotation, so on-field evidence can never
correct a mirrored belief mid-game (kidnap, wrong-side convergence). The
background scenery is not symmetric, and the disambiguator exploits that:

1. FAST corners + ORB descriptors per video frame, classified out-of-field
   by intersecting each ray with the ground plane / horizon.
2. An online landmark map in `{f}`: tracks triangulate when they accrue real
   parallax, otherwise promote as *bearing-only* landmarks at an assumed
   range (pose jitter is the same order as true parallax for most background
   structure, but bearing alone discriminates the mirror). Dynamic objects
   (crowd) are culled by a static-consistency test and miss-streak pruning.
3. Every frame, the same corners are associated against the map under the
   current pose *and its mirror*; the robust score difference accumulates
   into a forgetting log-likelihood ratio (LLR).
4. A sustained, dominant mirror preference requests a flip, which mirrors
   the filter belief `(x,y,yaw) → (−x,−y,yaw+π)` and negates the LLR.

Map building freezes whenever the pose is uncertain or the side is in doubt,
so a wrong-side excursion can never poison the map. Two flip paths exist:

- **Fair path**: deep LLR + enough mirror associations + mirror dominance +
  *coverage fairness* (own pose must see a comparable number of mapped
  landmarks — a degraded own pose is "no decision", not mirror evidence).
- **Blind-own escape**: when the wrong-side pose stares at territory the map
  never covered, coverage fairness can never be satisfied, so near-clamp LLR
  with an essentially blind own side (≤1 association) and steadily matching
  mirror qualifies over a longer leaky streak. On the recorded false-flip
  episode this streak peaks at 7 (threshold 40); on a genuine mirror lock it
  passes within ~2 s.

Verified by injected kidnaps (`KIDNAP_T=<s>`): detection + flip in ~7–15 s,
with re-convergence to <0.15 m within a second of the flip.

Note the out-of-field map is *not* part of the filter state — it is a
side-channel owned by `SideDisambiguator`, and its only coupling back to the
estimator is the discrete 180° flip. Outlier rejection therefore happens at
association and map-maintenance time rather than in a measurement update.

## Reading the camera panel

Data association is what the camera panel exists to debug, so both landmark
streams are colour-keyed by what the estimator did with each observation
(key drawn bottom-left; `k` hides it):

| Colour | Meaning |
| --- | --- |
| green | associated — YOLO detection, out-of-field corner (`o`) or map landmark (`[]`, with a line to its corner showing the residual) |
| amber | seen but unclaimed — landmark predicted in FOV that nothing matched, or a YOLO detection below the confidence threshold |
| red | rejected — corner gated to a landmark but beaten by clutter or by a closer corner (`x`), landmark culled this frame, or a usable YOLO detection outside the association gate |
| yellow | corner that matches the **mirrored** pose only: the wrong-side evidence the LLR accumulates |
| blue / cyan | corner growing a candidate track / out-of-field and unclaimed |
| mauve | landmark whose predicted bearing is too smeared to discriminate the mirror (`maxTangentSigma`) — excluded from matching *and* from scoring |
| grey | not in play: outside the FOV margin, or a YOLO class that is not a mapped landmark |

YOLO boxes keep their per-class colour so the class stays readable; the
status rides on the measurement dot, the label and the line thickness. The
thin rectangle inset from the border is `visibleMargin`: a landmark counts as
predicted-visible (and so may score and accrue misses) only inside it.

## Evaluation and ground truth

`data2/` carries OptiTrack mocap of the robot (~120 Hz) alongside the sensor
log; the frame alignment (field x = mocap y, field y = −mocap x, constant
yaw offset) lives in the `mocaptruth` namespace of `fieldLocalisation.cpp`.
Truth is evaluation-only — never fed to the estimator. The residual yaw
"error" on `data2` is a −5.3° *constant* with 1.1° scatter, attributable to
either the truth yaw-offset calibration or an unmodelled camera yaw mount
bias; the filter itself is internally consistent.

## Running it

```bash
./mcha4400 cmake --build nubots/build --target a2
./mcha4400 bash -c 'cd /workspace/nubots && ./build/a2 -r -e data2'   # headless: CSV/PNG/mp4 to out/
./mcha4400 bash -c 'cd /workspace/nubots && ./build/a2 -r data2 -i=1' # interactive viewer (note -i=1, not -i 1)
```

Useful environment switches: `NO_MP4=1` skips the video render on export
(fast tuning runs), `KIDNAP_T=<seconds>` mirrors the state mid-run to test
side recovery, `VIEWER_DUMP=1` renders viewer frames to PNGs without a
display. The exported `out/field_localisation.csv` contains the full state,
1-sigma bounds, baseline comparison, mocap truth and side-evidence per frame.
