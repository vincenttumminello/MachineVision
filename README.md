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

1. Open this repository in VS Code.
2. Install the **Dev Containers** extension if you haven't.
3. `F1` → **Dev Containers: Reopen in Container**.

`devcontainer.json` is pinned to the prebuilt `mcha4400:latest` image so
reopening never triggers the slow OpenCV-from-source build — rebuild deliberately
with `.devcontainer/build.sh` after editing the Dockerfile. VS Code mounts this
folder at `/workspace` and installs clangd + CMake Tools inside. IntelliSense, build, and debug all run against the
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
uncertainty, and it beats that baseline on both ground-truthed recordings:

| recording | truth | ours | NUbots baseline |
|---|---|---|---|
| `data2` (lab, OptiTrack) | mocap, 2681 samples | **0.093 m / 5.41°** | 0.506 m / 11.33° |
| `data4_webots` (sim, two real falls) | simulator, 971 shared samples | **0.115 m / 3.82°** | 0.265 m / 5.07° |

Over all 1779 of its own frames — including both eight-second falls, where the
baseline reports nothing — `data4_webots` comes to 0.135 m / 8.55°. Updates cost
~1.0 ms each on a laptop CPU (max 3.2 ms on `data2`), and 54/54 unit tests pass.

## State and belief representation

The state is 18-dimensional (`SystemLocalisation.h`):

| indices | meaning |
|---------|---------|
| 0–2     | torso position `rBFf` in the field frame `{f}` [m] |
| 3–6     | attitude quaternion `q` (w, x, y, z), `Rfb = quat2rot(q)` |
| 7–9     | body-fixed linear velocity `vBb` [m/s] |
| 10–12   | body-fixed angular velocity `omegaBb` [rad/s] |
| 13–15   | gyroscope bias `bGyro` in `{b}` [rad/s] |
| 16–17   | camera-mount attitude bias (roll, pitch) [rad] |

Pose in `{f}`, velocity in `{b}`: the Fossen vehicle-dynamics convention
`(eta, nu)` rather than the strapdown-INS one. The velocity states are a random
walk, and "the robot keeps walking forward at this speed" is a far better model
of a turning robot than "its field-frame velocity vector is constant" — the
latter needs a PSD sized for the turn even when walking straight. Every velocity
measurement also arrives body-fixed (the gyroscope directly, the walk odometry as
a body-frame finite difference), so `h(x)` is the identity rather than `Rfb'`. A
pleasant consequence: the 180° field mirror leaves both velocity blocks alone
(`v_b' = (Rz(π)Rfb)' Rz(π) v_f = v_b`), so `mirrorState()` stays a
position-and-quaternion operation.

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

## Prediction: nothing is a known input

The process model is autonomous — the rigid-body kinematics plus a random walk
on the rates — and is integrated through an SDE with RK4
(`SystemEstimator.cpp`):

```
d(rBFf)/dt = Rfb*vBb      dq/dt = 0.5*Xi(q)*omegaBb
d(vBb)/dt  = dw_v         d(omegaBb)/dt = dw_omega
d(bGyro)/dt = dw_b        d(deltaC)/dt  = dw_c
```

The gyroscope and the walk-engine odometry used to be a known *input* to this
model. They are now **measurements of the velocity states**
(`MeasurementBodyRates.h`), which is the point of carrying `vBb` and `omegaBb`:
an input asserts its value as truth, so its noise can only be expressed as
process noise on whatever it drives and any bias it carries is unmodelled,
whereas a measurement carries its own sigma, can be gated or suppressed, can be
contradicted by the rest of the belief, and can have a bias that is itself a
state.

- **`MeasurementGyroscope`**: `y = omegaBb + bGyro + v`, σ = 0.02 rad/s
  (`SIGMA_GYRO`). It runs unconditionally, including through a fall — it is the
  one sensor that measures a topple honestly, whatever the posture.
- **`MeasurementBodyVelocity`**: `y = vBb + v`, from finite-differencing the
  odometry stream `Htw` (`twistFromOdometry`) at σ = 0.15 m/s
  (`SIGMA_ODOM_VEL`) — loose but real information, present as evidence and never
  as truth. Suppressed while not upright, where the same class supplies a
  **zero-velocity update** instead (σ = 0.02 m/s lying still, 0.30 m/s while
  toppling or being levered upright).

The odometry's *attitude* is not used at all. The walk-engine attitude slips
badly while turning — on `data2` the odometry alone loses ~150° of yaw by
t = 40 s, mocap-verified — while the gyroscope tracks every turn and only drifts
at a smooth bias-like rate. That drift used to be removed by a heuristic that
averaged "quiet" samples; the bias is now a state, made observable because the
gyroscope sees the sum while the landmarks pin `omegaBb` through the attitude
they constrain. It is unobservable to the upstream Mahony filter, whose bias
integrator is driven by the cross product of two near-vertical vectors and so has
no component about the vertical — which is precisely the axis that turns into
heading drift. On `data2` it converges to `[+0.31, +2.40, −0.78] deg/s` at
σ = 0.09 deg/s, and the old heuristic's estimate of the z bias was ~0.75 deg/s.

Process noise is a diagonal PSD (`Parameters` in `SystemLocalisation.h`). The
rate states now carry the dominant terms — 0.35 m/s/√s on `vBb` (a walking robot
changes forward speed over a step) and 1.50 rad/s/√s on `omegaBb` — while the
pose PSDs (0.02 m/√s horizontal, 0.01 rad/√s attitude) are a floor against
collapse in weakly-observable directions rather than the main source of growth:
uncertainty reaches position by integrating velocity uncertainty, which is where
the real ignorance lives. The gyroscope bias walks at 2e-4 rad/s/√s (slow thermal
drift only, so vision can average it out over many frames) and the camera bias at
3e-4 rad/√s. The tuning target is *consistency*: the mocap truth lies inside the
3-sigma position bound on 98.7% of `data2` samples (see *Consistency* below for
the one direction where the belief is not consistent, and why that is a bias
rather than a tuning failure).

### Event ordering

`predict()` refuses to integrate backwards. A negative `dt` is not merely
inaccurate: RK4 runs with a negative step and the process-noise square-root
information goes as `1/(σ√dt)`, so the belief turns to NaN on the spot and every
subsequent measurement inherits it. The old guard was an `assert`, which does
nothing — `CMakeLists` defines `NDEBUG` for every non-Debug build and Release is
the default. An out-of-sequence event now leaves the belief and the clock alone
and is counted.

Rejecting is not the same as handling it: the measurement that follows is applied
at the filter's current time rather than its own, which is a real modelling error
whose size scales with the lag. `backwardPredicts()` and `maxBackwardDt()` exist
so that error is visible rather than assumed absent, and the run prints a warning
when it happens. Doing better needs an out-of-sequence measurement update, which
this is not. All four recordings report zero rejections, as an offline replay of a
correctly ordered stream should.

## Initialisation: first-frame grid solve + start-half prior

No baseline pose is used. `solveInitialPose()` (`fieldLocalisation.cpp`)
grid-searches (x, y, yaw) on the first usable vision frame (z/roll/pitch come
from kinematics), scoring each cell with the robust landmark log-likelihood
(~170 ms once). On-field landmarks are invariant under the field's 180°
rotation, so the global maximum has an equally-likely mirror; the
GameController-style start-in-own-half prior (`ownHalfXSign`) picks the side.
The filter then starts from a deliberately loose prior and sharpens
recursively.

The remaining states start where a robot at kick-off is: velocity at zero with a
prior wide enough to cover a walk if it is not (0.30 m/s, 0.50 rad/s), and the
gyroscope bias at zero with a prior covering a few deg/s, which is the scale of
the drift it exists to absorb. The four attitude entries all carry the loose yaw
figure (0.25 per component — a body rotation of `s` maps to `0.5*s` on the
components), because yaw uncertainty is not separable across them; the tight
roll/pitch prior is re-established within a frame by gravity.

This is the one place the side is decided in the shipped configuration, and it is
a prior that is true exactly once, at kick-off. On `data3_webots` it lands on the
mirrored half — the run ends 4.4 m and 164° from that log's NUbots baseline with
no flips — and nothing downstream corrects it (see *Side disambiguation*).

## Measurement models

Applied per vision frame through `system.process()` (which also drives the
optional Gaussian-mixture hypothesis bank):

- **Field landmarks** (`MeasurementFieldLandmarks`): YOLO detections arrive
  as calibrated unit rays in the camera frame (the lens model — equidistant
  fisheye on the robot, rectilinear under webots — is already applied upstream),
  classes = goal posts and L/T/X line intersections.
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
  and pitch. Gated on the *specific force* rather than on posture:
  `| ‖a‖ − g | < 3 m/s²`. That is the actual validity condition — true of a robot
  lying still, false in free fall or on impact — and a better test than "upright"
  while walking too (455 frames of `data2` fail it).
- **Kinematic height** (`MeasurementKinematicHeight`): torso height from the
  leg kinematic chain observes z. The one model a fall genuinely invalidates,
  so it is the one suppressed by posture.
- **Body rates** (`MeasurementGyroscope`, `MeasurementBodyVelocity`): see
  *Prediction* above. These go in before the no-detections gate, because neither
  has anything to do with whether YOLO found something — and a fallen robot's
  camera is in the carpet, so those are exactly the frames where an unmeasured
  velocity state would integrate the pre-fall gait straight off the field.
- **Quaternion norm** (`MeasurementQuaternionNorm`): unit-norm pseudo-measurement.
  The four attitude states carry three degrees of freedom and `quat2rot`
  normalises, so `|q|` is invisible to every other model; without this the MAP
  Hessian is singular along it and the Newton step has a flat direction to wander
  down.
- **Field lines** exist (`MeasurementFieldLines`) but ship disabled: their
  per-frame errors are strongly correlated and multi-line capture degraded
  accuracy on the recorded data.

## Falls (`FallDetector`, posture gate)

The starting position was that every measurement model above is an upright-robot
model. Gravity assumes the accelerometer reads gravity; kinematic height assumes
the support leg reaches the ground; the landmark model assumes predicted bearings
land inside a 0.35 rad gate; the side disambiguator assumes the pose it
triangulates background corners from is roughly right. A fall appears to break all
four at once, and none of them degrade gracefully — the accelerometer reads ≈0 in
free fall and 20–40 m/s² on impact against a 1 m/s² noise model (a 10–30σ pull on
roll and pitch), and a handful of landmarks that happen to line up under the wrong
attitude will happily *shrink* the covariance around a pose that is simply wrong.
Only one of the four turned out to be a genuine model failure; the rest of this
section is how that got sorted out.

`FallDetector` builds a posture timeline for the log, preferring the robot's own
stability flags (`SensorLog::stability`, parsed from any `*.Stability` message)
and falling back to the tilt of the smoothed accelerometer when the log has none.
Smoothing is essential: ordinary walking spikes the *raw* accelerometer past 90°
of apparent tilt for single samples, so nothing instantaneous is usable. The
kinematic chain is deliberately not trusted for this — in both recordings it
reports a near-upright torso at 0.44 m even through a 34° lean.

While the posture is anything other than upright:

- **Kinematic height is suppressed**, and gravity falls back on its
  specific-force gate. Landmarks and out-of-field keep running: the gate is
  per-model now, and the reasoning for that is the subsection below — the
  all-or-nothing version was never really about the measurement models.
- **Prediction still runs.** It previously only ever happened inside
  `Event::process`, so a frame with no detections advanced neither the state nor
  the clock. A face-down fall produces exactly that, and the filter would emerge
  holding its pre-fall mean at its pre-fall covariance — confidently wrong rather
  than honestly uncertain. `SystemLocalisation::predictAll` fixes that, for every
  hypothesis when the bank is live.
- The **odometry velocity measurement is dropped and a zero-velocity update takes
  its place**, for the *whole* episode. Walk-engine odometry describes the gait
  the engine believes it is executing, which on the ground is fiction and during
  a getup is a scripted flail that is not locomotion. That is a statement about
  whether the signal means anything, and it does not become true again after some
  number of seconds. Leaving the velocity unmeasured instead is not the neutral
  choice it looks like — it says the robot may still be travelling at whatever it
  was doing when it fell, which is the one thing it is certainly not doing; on
  `data4_webots` that let the error compound 0.12 → 0.62 → 1.76 m over two falls
  and never recover. The gyroscope is kept throughout, because it measures the
  topple for real.
- Process noise switches to the `*Disturbed` PSDs (1.00 m/s/√s on velocity,
  3.00 rad/s/√s on the body rate, 0.20 m/√s and 0.20 rad/√s on the pose)
  **for the first 2 s only** (`params.disturbedWindow`), and then stands back
  down. What a fall does to the pose is a bounded event, not a diffusion: the
  torso moves while it topples and while it is levered upright, and in between it
  lies still. Running the disturbed PSDs for the whole window instead made the
  belief's width report how long the robot had been down rather than how far it
  could have gone. Across the two real falls in `data3_webots` the NUbots baseline
  moves 0.0 m and 0.6 m, so the displacement is event-sized and the one-shot
  inflation below is where it belongs.

Validity and diffusion are deliberately **not** the same switch
(`setPosture(upright, disturbedFor)`), because they expire differently. Tying
both to the 2 s window is what let a long fall integrate the getup's odometry at
nominal confidence past it, marching the estimate away with a covariance too
tight for the association gate to recover from.

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
untouched (`data2` was bit-identical when it landed) and the surprisal score
still decides what actually associates.

Separately, `TKfromThetaTemplated` now saturates `|cos(pitch)|` at 1e-3. The
roll-pitch-yaw rate transform is singular at pitch = ±90°, which a forward or
backward fall passes straight through; unguarded, the infinite Jacobian propagates
into the predicted covariance and hands the Newton update a NaN prior, poisoning
the filter for the rest of the run rather than just for the fall. The clamp never
binds below 89.94° of pitch, so upright behaviour is bit-for-bit unchanged.

`data` and `data2` contain no fall — the torso never leaves 0.43–0.44 m and tilt
peaks at 34°. Both webots recordings **do**, and the accelerometer fallback finds
two apiece: t = 54.0–62.1 s and t = 74.6–82.4 s on `data3_webots` (the video
confirms both — the camera is looking at open sky through the first and buried in
the carpet through the second), t = 30.3–38.3 s and t = 71.2–78.8 s on
`data4_webots`. The `data3_webots` figures are 6.2 s earlier than this README used
to quote, following the clock alignment described below. That is the detector
firing on genuine topples, which the injected-fall harness below cannot
demonstrate.

`data4_webots` is the case with ground truth through the falls, and the filter
recovers from both: 0.135 m / 8.55° over all 1779 frames, on 971 of which the
NUbots baseline also reports.

`FALL_T=<s> FALL_DURATION=<s>` forces the posture to fallen over a window so the
suppress → coast → inflate → recover path can be exercised on the mocap
recording. On `data2` at t = 40 s:

| fall length | σ_xy at recovery | σ_yaw at recovery | RMSE vs mocap |
|---|---|---|---|
| none | — | — | 0.093 m / 5.41° |
| 3 s | 0.51 m | 60.0° | 0.093 m / 5.41° |
| 12 s | 0.50 m | 60.0° | 0.104 m / 5.41° |
| 30 s | 0.50 m | 60.0° | 0.145 m / 5.28° |

The recovery belief describes the event rather than the clock: it is now
essentially the inflation itself (+0.50 m, +60°), because the zero-velocity update
holds `vBb` at zero through the blind window and the pose PSDs are a floor, so
almost nothing diffuses. Earlier designs grew without bound here — 1.48 m and 134°
after 12 s, ≈2.2 m and 200°+ after 30 s, for a robot that had not moved since it
landed.

Note what this harness does and does not show. The robot is really upright and
walking throughout, so the zero-velocity update it triggers is a false
measurement — which is exactly why the 30 s row is the worst of the four. It
demonstrates that the filter survives and reconverges after a blind window, not
that the fall response is right for a real topple; for that, see the two real
falls in `data4_webots` above.

Two ablation switches exist to measure the design rather than assume it.
`FALL_GATE=off` makes the whole run read as upright — no suppression, no
zero-velocity update, no recovery inflation — and `FALL_INFLATE=<m>,<deg>` changes
(or with `0,0` removes) what recovery hands back. Run against `data4_webots`,
where both falls are real and ground truth covers them:

| configuration | RMSE vs simulator truth |
|---|---|
| as shipped | 0.135 m / 8.55° |
| `FALL_INFLATE=0,0` | 0.128 m / 8.54° |
| `FALL_GATE=off` | 0.117 m / 8.45° |

**Turning the fall handling off is currently the more accurate configuration on
the only recording that can test it.** That is worth stating plainly rather than
burying: the posture response does not pay for itself in RMSE here. What it buys
is a failure mode this recording does not contain — a robot that lies still after
it lands produces odometry that is wrong but small, whereas a live getup produces
a scripted flail that the walk engine reports as locomotion, and integrating that
at nominal confidence is what marches the estimate off the field with a covariance
too tight to recover. The 18 mm the gate costs on `data4_webots` is the premium
paid for that; whether it is worth paying needs a recording of a real getup.

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

Attitude is now the quaternion block of the state table above, with
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

On `data3_webots` the attitude tracks continuously through both falls
(pitch −8.8° → −48.0° → back, yaw 179.4° → 179.5°) with σ_yaw rising to 17°
during the blind-ish window rather than the filter being blind outright, and
`data2` was unchanged against mocap when the change landed.

**Still open.** The tracked pitch peaks at 48° through a fall where the camera is
looking at open sky, so the true attitude is nearer 90° — the filter follows the
fall in the right direction but under-shoots it. Tuning that (accelerometer noise,
the quasi-static threshold, how much the out-of-field corners should pull attitude)
needs ground truth *during* a fall. `data3_webots` does not carry any: its
`RobotPoseGroundTruth` channel is present in every `RawSensors` message but has
`exists == false` throughout. `data4_webots` does carry the channel and does fall
twice, so it is the recording to tune this against — its ground truth spans the
whole 87.5 s including both falls.

## Field dimensions (`--field`)

Two different fields are in play, and they are not close. `data` and `data2` were
recorded on the **lab field** (6.8 × 5 m); the webots worlds use the full kid-size
**RoboCup field** (9 × 6 m), which also has a wider goal (2.6 m vs 1.95 m), a
different penalty-mark distance and a 1.0 m border strip instead of 0.38 m. Both
are NUbots configs — `SoccerConfig/data/config/FieldDescription.yaml` is the lab
one this project's `FieldDimensions` defaults were transcribed from, and
`SoccerConfig/data/config/webots/FieldDescription.yaml` is the one the simulator
and its `RobotPoseGroundTruth` are produced against.

Replaying a webots recording against the lab map is not a small error: every
landmark sits metres from where the robot actually sees it, and the filter absorbs
the difference as pose error. On `data4_webots` it moved the initial solve from
(1.82, −2.88) to (2.70, −3.35) — the truth's first sample is (2.96, −3.38) — and
took position RMSE against truth from 0.955 m to 0.753 m at the time it landed
(both figures predate later fixes; the run is at 0.135 m now), while bringing the
torso height error to within a centimetre.

`--field=lab|webots` names one explicitly. Otherwise it follows the camera
calibration, since a recording made through the simulated camera was made in the
simulated world.

## Ground truth (two sources, two clocks)

Evaluation reference only — neither stream ever reaches the estimator.

| recording | stream | what `rBFf` is |
|---|---|---|
| `data`, `data2` | `message.input.MotionCapture` (OptiTrack) | the **marker body**, ~6 cm above the torso origin, after undoing the capture-volume rotation and yaw extrinsic |
| `data4_webots` | `message.localisation.RobotPoseGroundTruth` (`Hft`) | the **torso origin** itself, already in the field frame |
| `data3_webots` | — | none: the `RawSensors` copy of `RobotPoseGroundTruth` is present on all 13082 messages with `exists == false` and uninitialised `Hft` |

The simulator path is deliberately thin. `Hft` is already the torso pose in the
frame the estimator works in — verified against that log's own NUbots baseline,
which tracks it to ~0.2 m over a whole run — so there is no volume alignment to
undo, no marker offset and no yaw extrinsic. Anything more would be inventing a
correction the simulator has already applied. `TruthSource` records which stream a
run used, because the height comparison means different things for each (0 is the
target for the simulator, ≈−0.06 m for markers).

**Clocks.** The log mixes two timestamps: `Sensors`, the vision messages and the
field-line points carry a payload timestamp, while `WalkState`, the stability
stream, the NUbots baseline, motion capture and the ground truth carry only the
envelope timestamp NUClear stamped on receipt. On a real robot those are the same
wall clock and the mix is harmless, which is why it went unnoticed. Under webots
the payload timestamp is **simulation** time — `data4_webots` stamps its `Sensors`
messages `1970-01-01T00:15:44.824Z` — so the two clocks sit 1.785 × 10⁹ s apart and
every envelope-stamped stream lands nowhere near the frames it should align with.
Nothing detected it; the comparisons just never matched.

`Sensors` carries both stamps, so `SensorLog` measures the median offset and
shifts the envelope-stamped streams onto the payload clock, reporting when it
does. The threshold is 1 s, so the two real-robot logs (`data`, `data2`, both
wall-clock throughout) are untouched and bit-for-bit unchanged. `data3_webots`
does get shifted, by a much smaller −6.2 s — small enough to pass for ordinary
latency, large enough to move its posture windows and its baseline comparison.

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

The score in step 3 charges **misses as well as hits**: every landmark predicted
well inside the image and not matched costs `missPenalty` (0.5 nats, `SIDE_MISS`).
This is equation (13)'s `−4|U|log|Y|` null-hypothesis term, one constant per miss
because these landmarks are points rather than four-corner tags. Summing only over
accepted associations rewarded a hypothesis for predicting a lot and never charged
it for being wrong about most of it, which is enough to saturate the LLR on
evidence that is entirely an artefact of where the map was built: on
`data4_webots` a mirror that predicted 302 landmarks and matched 70 (23%)
outscored an own hypothesis that predicted nothing at all, and `freezeLlr` then
locked map building out of ever covering the direction that would settle it. The
constant is calibrated rather than derived — 0.4–0.6 works on all three
recordings, 0.2 leaves `data4_webots` saturated, 0.8 drives `data2` negative and
collapses its map from 500 landmarks to 265 through the same freeze.

Map building freezes whenever the pose is uncertain or the side is in doubt,
so a wrong-side excursion can never poison the map. Two flip paths exist:

- **Fair path**: deep LLR + enough mirror associations + mirror dominance +
  *coverage fairness* (own pose must see a comparable number of mapped
  landmarks — a degraded own pose is "no decision", not mirror evidence).
- **Blind-own escape**: when the wrong-side pose stares at territory the map
  never covered, coverage fairness can never be satisfied, so near-clamp LLR
  with an essentially blind own side and steadily matching mirror qualifies over
  a longer leaky streak. Two guards keep that from firing on an honest turn.
  The own hypothesis must **predict something and fail to match it**
  (`flipBlindMinVisibleOwn`), which is a contradiction, rather than predict
  nothing, which is an absence: `visOwn == 0` says the map does not cover where
  own is looking, and a correctly localised robot produces that every time it
  turns to face new scenery. And the escape is refused within
  `blindTurnTolerance` (60°) of a **net 180° turn** since the map last confirmed
  the own side, because against a background that is itself symmetric — the
  webots stadium, both ends alike — turning around in place and being mirrored
  produce the same observation exactly. Without those, `data4_webots` flipped a
  pose that was correct to 2 cm after the robot physically turned 164.6°, and
  finished the run at 0.75 m / 80° against truth instead of 0.12 m / 8.5°.

The escape used to be verified by injected kidnaps (`KIDNAP_T=<s>`): a flip within
~7–15 s of the displacement and re-convergence to <0.15 m within a second of it.
A kidnapped robot is carried rather than turned under its own gyroscope, so the
turn gate is not what stands in the way. **The miss penalty is.** As shipped,
`data2` no longer recovers from an injected kidnap at all:

| run | outcome |
|---|---|
| `KIDNAP_T=40` | no flip; 1.869 m / 139.7° over the run |
| `KIDNAP_T=60` | no flip; 1.656 m / 117.5° |
| `KIDNAP_T=40 SIDE_MISS=0` | flip at t = 59.9 s; 0.870 m / 75.6° (the residual is the 20 s spent mirrored before it fires) |

The likely mechanism, not yet confirmed: charging misses is symmetric, but after
a kidnap the two sides are not. The true pose looks back over mapped territory and
so predicts a great many landmarks, matching a fraction of them, while the
kidnapped pose predicts and misses far less — so the penalty falls hardest on the
hypothesis that is right and drags the ratio back below the flip threshold. The
fix is not a knob to nudge: dropping the penalty re-saturates `data4_webots`,
which is exactly what it was added for. Recorded here as an open regression.

**Where the rest of the evidence stands.** No recording flips of its own accord,
and on `data2` and `data4_webots` the LLR sits at its +40 clamp in favour of the
pose the filter holds, which is the correct one. On `data3_webots` it also
reads +40 — for
a pose that is mirrored (4.4 m and 164° from that log's NUbots baseline). The most
likely explanation is that the map was built under the mirrored pose from the
first frame and is perfectly self-consistent in it, which no map-based test can
escape. Note also that on a venue whose background is 180°-symmetric, this
evidence is not merely weak but actively misleading, and the side is really being
held by the start-half prior at initialisation.

Note the out-of-field map is *not* part of the filter state — it is a
side-channel owned by `SideDisambiguator`, and its only coupling back to the
estimator is the discrete 180° flip. Outlier rejection therefore happens at
association and map-maintenance time rather than in a measurement update.

**The map has almost no depth, by construction.** Background structure rarely
accrues parallax above pose jitter, so the overwhelming majority of promotions
are *bearing-only*: `fitFar()` parks them at `assumedRange` (6 m) along the
measured bearing with a 3σ radial spread of 9 m. On `data2` that is 6210
bearing-only promotions against 39 triangulated, of which 581 are later upgraded
by real parallax, leaving a live map (capped at 500) of 201 bearing-only and 299
triangulated at a median 3σ of 3.3 m. `data4_webots` never earns that parallax and
ends 386 / 99 at a median 3σ of 9.1 m. Two consequences worth knowing before
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

The run summary tallies what happens to every in-view prediction, which is how
much of the map is doing work: on `data2`, 45.5% associate, 26.0% are charged as
misses, 27.8% are dropped as *ambiguous* (predicted bearing too smeared to
discriminate the mirror, `maxTangentSigma`) and 0.7% sit too near the border to
count. The ambiguous share runs 20–32% across the three recordings — and it is
**not** the bearing-only landmarks that waste it. They are 40% of the `data2` map
and 0.4% of its ambiguity, while carrying 96–99% of all associations on every
recording. The `assumedRange` prior is not what costs the map its discrimination.

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

Default **off**, as a compile-time constant in `runFieldLocalisation` rather than
a flag: initialisation already fixes the side from the known start half (see
above), so on the recorded data the bank only adds a brief two-component window
(~15 s until the map prunes the mirror) before collapsing to the
single-hypothesis path, at accuracy indistinguishable from it when it was last
measured. It is the mechanism to enable for a robot that can start on an unknown
side, or be displaced without a GameController signal — the case none of the
recorded runs exercise, so the resolution path rests on unit tests rather than on
a natural mid-game flip. It shares the disambiguator's clamped own-minus-mirror
score, so the `KIDNAP_T` evidence it used to lean on is subject to the same open
regression noted above.

## Reading the camera panel

Data association is what the camera panel exists to debug, so both landmark
streams are colour-keyed by what the estimator did with each observation
(key drawn bottom-left; `k` hides it):

| Colour | Meaning |
| --- | --- |
| green | associated — YOLO detection, out-of-field corner (`o`) or map landmark (`[]`, with a line to its corner showing the residual) |
| amber | seen but unclaimed — landmark predicted in FOV that nothing matched (each of these now costs its hypothesis `missPenalty`), or a YOLO detection below the confidence threshold |
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

## Consistency, and the residual yaw offset

`data2/` carries OptiTrack mocap of the robot (~120 Hz) alongside the sensor
log; the frame alignment (field x = mocap y, field y = −mocap x, constant
yaw offset) lives in the `mocaptruth` namespace of `fieldLocalisation.cpp`.
Truth is evaluation-only — never fed to the estimator.

Position is consistent: 98.7% of `data2` samples fall inside the 3-sigma bound.
Yaw is not, and the shape of the failure says why. The residual is a −5.3°
*constant* with 1.1° scatter, so the filter's scatter is right and its mean is
displaced by a bias — attributable to either the truth yaw-offset calibration or
an unmodelled camera yaw mount bias, neither of which the state can absorb (the
camera-bias states are roll and pitch only). A 5° systematic against a sub-degree
belief means the 3-sigma yaw bound covers 5% of samples, and that number should be
read as "there is an uncorrected offset here", not as an inconsistent filter.
It is also the whole of the 5.41° yaw RMSE: √(5.28² + 1.18²) = 5.41.

## Running it

```bash
./mcha4400 cmake --build nubots/build --target a2
./mcha4400 bash -c 'cd /workspace/nubots && ./build/a2 -r -e data2'   # headless: CSV/PNG/mp4 to out/
./mcha4400 bash -c 'cd /workspace/nubots && ./build/a2 -r data2 -i=1' # interactive viewer (note -i=1, not -i 1)
./mcha4400 bash -c 'cd /workspace/nubots && cmake --build build --target tests && ./build/tests'
```

Recordings: `data`, `data2` (lab, `data2` has mocap), `data3_webots`,
`data4_webots` (simulator, both with falls; `data4_webots` has ground truth).

Command-line options (OpenCV's parser wants `-i=1`, not `-i 1`):

| flag | meaning |
|---|---|
| `-r`, `--robocup` | run field localisation on the given data directory |
| `-e`, `--export` | write CSV/PNG/mp4 to `out/` |
| `-i=0\|1\|2` | interactivity: none, last frame, every frame |
| `-v=0..3` | log verbosity: silent; one line per event; plus optimiser summary and per-hypothesis detail; plus per-iteration trace |
| `--lens=<name>` | camera calibration to replay with (default: from the frame size) |
| `--field=lab\|webots` | field the recording was made on (default: follows the camera) |
| `-h` | help, including the available lens calibrations and field sizes |

Environment switches, all defaulting to the shipped behaviour:

| variable | effect |
|---|---|
| `NO_MP4=1` | skip the video render on export (fast tuning runs) |
| `VIEWER_DUMP=1` | render viewer frames to PNGs without a display |
| `VIEWER_3D=1` | open the viewer with the 3D pane on the right |
| `KIDNAP_T=<s>` | mirror the state mid-run to test side recovery |
| `FALL_T=<s>`, `FALL_DURATION=<s>` | force the posture to fallen over a window |
| `FALL_GATE=off` | replay with no posture handling at all |
| `FALL_INFLATE=<m>,<deg>` | change (or with `0,0` remove) the recovery inflation |
| `GYRO_BIAS=off` | freeze the gyroscope bias state at zero |
| `SIGMA_GYRO=<rad/s>`, `SIGMA_ODOM_VEL=<m/s>` | rate-measurement noise sweeps |
| `SIDE_MISS=<nats>` | the disambiguator's per-miss penalty |

The exported `out/field_localisation.csv` contains the full state, 1-sigma
bounds, baseline comparison, ground truth and side-evidence per frame.
