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
  Rays are pre-gated (20°) and associated against the known field map at the
  prior mean by **surprisal nearest neighbour**: pairs are ranked and accepted
  by surprisal in the tangent plane of the predicted bearing, relative to the
  clutter crossover, rather than by raw angle. The likelihood is a robust
  inlier-Gaussian + uniform-clutter mixture on the unit sphere (σ = 0.25 rad),
  so gross mis-detections are absorbed rather than fought.
  The inlier weight is **per-detection, taken from the YOLO confidence**
  (w ≈ confidence, capped at 0.95): confidence is a statement about whether the
  box is a true positive, which is what the mixture weight means — inflating σ
  instead would claim the landmark is certainly real but poorly measured. As
  w → 0 the term flattens towards clutter, contributing to neither the gradient
  nor the Hessian, so weak detections cannot sharpen the posterior.
- **Gravity** (`MeasurementGravity`): accelerometer direction observes roll
  and pitch.
- **Kinematic height** (`MeasurementKinematicHeight`): torso height from the
  leg kinematic chain observes z.
- **Field lines** exist (`MeasurementFieldLines`) but ship disabled: their
  per-frame errors are strongly correlated and multi-line capture degraded
  accuracy on the recorded data.

## Falls (`FallDetector`, posture gate)

Every measurement model above is an upright-robot model. Gravity assumes the
accelerometer reads gravity; kinematic height assumes the support leg reaches the
ground; the landmark model assumes predicted bearings land inside a 0.35 rad
gate; the side disambiguator assumes the pose it triangulates background corners
from is roughly right. A fall breaks all four at once, and none of them degrade
gracefully — the accelerometer reads ≈0 in free fall and 20–40 m/s² on impact
against a 1 m/s² noise model (a 10–30σ pull on roll and pitch), and a handful of
landmarks that happen to line up under the wrong attitude will happily *shrink*
the covariance around a pose that is simply wrong.

`FallDetector` builds a posture timeline for the log, preferring the robot's own
stability flags (`SensorLog::stability`, parsed from any `*.Stability` message)
and falling back to the tilt of the smoothed accelerometer when the log has none.
Smoothing is essential: ordinary walking spikes the *raw* accelerometer past 90°
of apparent tilt for single samples, so nothing instantaneous is usable. The
kinematic chain is deliberately not trusted for this — in both recordings it
reports a near-upright torso at 0.44 m even through a 34° lean.

While the posture is anything other than upright:

- The belief is **predicted and nothing else** — no gravity, kinematic-height,
  landmark or out-of-field update is applied.
- **Prediction still runs.** It previously only ever happened inside
  `Event::process`, so a frame with no detections advanced neither the state nor
  the clock. A face-down fall produces exactly that, and the filter would emerge
  holding its pre-fall mean at its pre-fall covariance — confidently wrong rather
  than honestly uncertain. `SystemLocalisation::predictAll` fixes that, for every
  hypothesis when the bank is live.
- The **twist input's linear velocity is zeroed** (`setDisturbed`). It is a finite
  difference of walk-engine odometry, which keeps describing the gait it believes
  it is executing while the robot is on the ground; the zero-order hold would
  otherwise carry the last pre-fall velocity across the whole event. The
  gyroscope-derived angular rate is kept, because it measures the topple for real.
- Process noise switches to the `*Disturbed` PSDs (≈0.40 m/√s in position,
  0.60 rad/√s in yaw) **for the first 2 s only**, and then stands back down. What
  a fall does to the pose is a bounded event, not a diffusion: the torso moves
  while it topples and while it is levered upright, and in between it lies still.
  Running the disturbed PSDs for the whole window instead made the belief's width
  report how long the robot had been down rather than how far it could have gone —
  1.48 m of position std after 12 s, for a robot that had not moved since it
  landed, and a yaw std of 134°, which is not a meaningful figure on a circle.
  Across the two real falls in `data3_webots` the NUbots baseline moves 0.0 m and
  0.6 m, so the displacement is event-sized and the one-shot inflation below is
  where it belongs.

On recovery the **mean is kept** and only the covariance is widened (+0.50 m in
position, +60° in yaw). A fall and getup translate the torso well under a metre,
so the pre-fall position remains the best estimate available. Re-running the
initial grid solve would be actively worse: it resolves field symmetry from the
known starting half, a prior that is true exactly once, at kick-off — mid-game it
would drag a correctly localised robot standing in the opponent half back across
the halfway line. The inflated position std also exceeds `SideDisambiguator`'s
`maxPosStd`, which freezes background-map building until the filter reconverges,
so no landmarks are triangulated from the recovering pose. With the bank live, the
mirror is re-seeded, since a fall is also an opportunity to have been turned around
without the landmarks being able to notice — but note that `useHypothesisBank`
defaults to `false`, so in the shipped configuration that re-seed never runs.

For the inflation to be worth anything, the association gate has to be able to
reach the widened belief. `MeasurementFieldLandmarks`' pre-gate is a *geometric*
cap on the residual angle, so it does not consult the covariance at all: at a
fixed 0.35 rad it silently bounded what the filter could ever recover from, and a
getup that turned the robot further than 20° put every predicted bearing outside
it no matter how much variance recovery handed back. The pre-gate now widens to
`gateYawScale` (2) yaw std devs, capped at `gateAngleMax` (1 rad). It takes
σ_yaw > 10° before that exceeds the nominal 0.35 rad, so ordinary operation is
untouched — `data2` is bit-identical at 0.107 m / 5.41° — and the surprisal score
still decides what actually associates.

Separately, `TKfromThetaTemplated` now saturates `|cos(pitch)|` at 1e-3. The
roll-pitch-yaw rate transform is singular at pitch = ±90°, which a forward or
backward fall passes straight through; unguarded, the infinite Jacobian propagates
into the predicted covariance and hands the Newton update a NaN prior, poisoning
the filter for the rest of the run rather than just for the fall. The clamp never
binds below 89.94° of pitch, so upright behaviour is bit-for-bit unchanged.

`data` and `data2` contain no fall — the torso never leaves 0.43–0.44 m and tilt
peaks at 34°. `data3_webots` **does**: the accelerometer fallback flags
t = 60.2–68.3 s and t = 80.8–88.6 s, and the video confirms both (the camera is
looking at open sky through the first and buried in the carpet through the
second). That is the detector firing on genuine topples on real data, which the
injected-fall harness below cannot demonstrate.

`FALL_T=<s> FALL_DURATION=<s>` forces the posture to fallen over a window so the
suppress → coast → inflate → recover path can be exercised where ground truth
exists. On `data2` at t = 40 s:

| fall length | σ_xy at recovery | σ_yaw at recovery | RMSE vs mocap |
|---|---|---|---|
| none | — | — | 0.107 m / 5.41° |
| 3 s | 0.77 m | 77.6° | 0.108 m / 5.40° |
| 12 s | 0.92 m | 78.3° | 0.110 m / 5.41° |
| 30 s | 0.91 m | 79.3° | 0.099 m / 5.63° |

The recovery belief now describes the event rather than the clock: it saturates
instead of growing without bound (the same three rows were 0.86 m/84.5°,
1.48 m/133.9° and ≈2.2 m/200°+ before). Note what this harness does and does not
show: the robot really is upright throughout, so it demonstrates that the filter
survives and reconverges after a blind window, not that suppression is the right
response to a real topple.

Two ablation switches exist to measure the design rather than assume it.
`FALL_GATE=off` replays straight through, applying every update; `FALL_INFLATE=<m>,<deg>`
changes (or with `0,0` removes) what recovery hands back. Both leave `data2`'s RMSE
at 0.107–0.111 m, because its fall is synthetic and the filter reconverges within
2–7 frames either way — the fall response is, on the available ground-truthed data,
unfalsifiable, and the numbers above are design arithmetic rather than evidence.

### Attitude is a quaternion (and why the gate is now per-model)

Suppressing *every* update while not upright was never really justified by the
measurement models being invalid — a robot lying still on the carpet gives a
perfectly good gravity vector, and the landmark and out-of-field models are plain
geometry. It was justified by the **state parameterisation**. Attitude used to be
roll-pitch-yaw, whose rate transform is singular at pitch = ±90°, and that is not
an edge case for a falling robot: it is on the trajectory of every topple. Passing
through it landed the state on the gimbal alias `(roll+180, 180−pitch, yaw+180)` —
the same rotation, so `fieldPose()` and the landmark models carried on working,
but every consumer reading `x(5)` as heading was then 180° out. Both real falls in
`data3_webots` did exactly this: stored attitude went from `(−0.0°, 8.7°, 178.7°)`
to `(541.3°, −162.1°, −361.3°)`, whose canonical form is `(1.3°, −17.9°, 178.8°)`.
The heading never moved; only the chart flipped.

The state is now `[x y z | qw qx qy qz | camBiasRoll camBiasPitch]` (nx = 9), with
`Rfb = quat2rot(q)` and `q̇ = ½·Ξ(q)·ω_b` — every entry of `Ξ` linear in `q`, so a
topple is an ordinary point. The cost is a fourth parameter for three degrees of
freedom, handled three ways: `quat2rot` normalises (so `|q|` is invisible to every
geometric model and cannot corrupt attitude), `MeasurementQuaternionNorm` supplies
the only information along that direction (without it the MAP Hessian is singular
there), and `SystemLocalisation::normaliseQuaternion` projects the mean back onto
the sphere after every predict and update. Attitude quantities that used to index
one element — process noise, the association gate's yaw variance, the recovery
inflation — go through `attitudeTangent`/`attitudeCovariance` instead; yaw
uncertainty about field z is a **rank-one** block on the quaternion states, not a
diagonal entry.

With that in place the posture gate is per-model rather than all-or-nothing:

- **Kinematic height** stays suppressed. It is the one model a fall genuinely
  invalidates — lying down, the chain still reports a near-upright 0.44 m torso.
- **Gravity** is gated on the *specific force* instead of on posture:
  `| ‖a‖ − g | < 3 m/s²`. That is the actual validity condition (true of a robot
  lying still, false in free fall or on impact) and a better test than "upright"
  while walking too.
- **Landmarks and out-of-field** keep running. Given the right attitude they are
  as valid face-down as standing, and a fall is exactly when they are needed: a
  robot that spins while toppling or getting up changes its heading, and only
  measurements taken during the event can catch it.

On `data3_webots` the attitude now tracks continuously through both falls
(pitch −8.8° → −48.0° → back, yaw 179.4° → 179.5°) with σ_yaw rising to 17°
during the blind-ish window rather than the filter being blind outright, and
`data2` is unchanged at 0.108 m / 5.40° against mocap.

**Still open.** The tracked pitch peaks at 48° through a fall where the camera is
looking at open sky, so the true attitude is nearer 90° — the filter follows the
fall in the right direction but under-shoots it. Tuning that (accelerometer noise,
the quasi-static threshold, how much the out-of-field corners should pull attitude)
needs ground truth during a fall, which `data3_webots` does not carry: its
`RobotPoseGroundTruth` channel is present in every `RawSensors` message but has
`exists == false` throughout.

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

**The map has almost no depth, by construction.** Background structure rarely
accrues parallax above pose jitter, so the overwhelming majority of promotions
are *bearing-only*: `fitFar()` parks them at `assumedRange` (6 m) along the
measured bearing with a 3σ radial spread of 9 m. On `data2` that is ~4300
bearing-only against ~30 triangulated. Two consequences worth knowing before
reading the 3D view or trying to reuse the map:

- Bearing-only landmarks lie on a **shell** around whichever camera position
  anchored them. That is the dominant feature of the 3D map and is expected,
  not a rendering bug — the run summary prints the split so it is not a
  surprise.
- The map can answer *"am I mirrored?"* — a binary question with a 180° margin
  that survives gross depth error — but not *"where exactly am I?"*. Feeding
  these landmarks in as position measurements would be wrong twice over: the
  assumed range biases the predicted bearing as soon as the camera translates,
  and the positions are themselves a function of past pose estimates, so
  treating them as independent evidence double-counts information and collapses
  the covariance. Using them metrically means putting them **in** the state with
  their pose cross-covariances, i.e. actual SLAM.

## Hypothesis bank (`SystemLocalisation`, off by default)

The disambiguator resolves the symmetry by a hard flip on a single Gaussian.
The alternative — a **weighted Gaussian mixture** carrying both the pose and
its 180° mirror (B-Human style) — also exists, behind `useHypothesisBank`.
The two are different answers to the *same* question and do not stack; the bank
is the substrate you would enable *instead of* the hard flip, and when it is on
the flip path is replaced by weight updates.

The key fact the design turns on: **on-field landmarks cannot separate the two
hypotheses.** Re-associated at its own pose, the mirror fits the mirror-partner
landmarks exactly as well as the true pose fits the originals, so the mixture
sits at a genuine 50/50 on landmark evidence alone (the earlier code only
collapsed it because association was shared from the representative — a bug the
mixture path now avoids by re-associating per component). What breaks the tie is
the **asymmetric out-of-field map**: each frame the disambiguator's clamped
own-minus-mirror score is folded into the weights (`addSideLogEvidence`), which
collapses the mirror once the background map favours one side, and a mid-game
kidnap re-seeds it (`spawnMirror`) for the weights to re-resolve. Output
stability at the 50/50 point comes from hysteresis in `setRepresentative` (the
reported pose does not flicker between a side and its mirror on numerical
noise).

Default **off**: initialisation already fixes the side from the known start
half (see above), so on the recorded data the bank only adds a brief
two-component window (~15 s until the map prunes the mirror) before collapsing
to the single-hypothesis path, at identical accuracy (0.107 m / 5.4°). It is
the mechanism to enable for a robot that can start on an unknown side, or be
displaced without a GameController signal — the case none of the recorded runs
exercise, so the resolution path is verified by unit tests and an injected
`KIDNAP_T` rather than by a natural mid-game flip.

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

The line joining an associated landmark to its corner is an **innovation, not
a frame-to-frame motion**: the box is where the stored landmark reprojects
through the *current pose estimate*, the circle is where the corner was
actually detected. Neither endpoint is ground truth, and the gap mixes three
error sources — landmark position error (large; see the depth note above),
pose error, and corner noise.

The 3D pane (`3`) uses the same colour key for the map. A triangulated
landmark whose 3σ ellipsoid is compact (< 1.5 m) gets the full three-ring
wireframe; a looser one, if associated in the current frame, gets a **bar
along the dominant 3σ axis** — "somewhere along here".

Bearing-only landmarks get a dot and nothing else. Their radial spread is not
an estimate: `fitFar()` fixes the range at `assumedRange` and the radial sigma
at half of it, so every bar would be identical in length and the pane would
fill with hundreds of interchangeable rays — noise dressed up as measurement.
The HUD line and the `map depth:` summary carry that information instead.

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
