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
