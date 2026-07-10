# syntax=docker/dockerfile:1
# MCHA4400 toolchain image
# Replaces the manual "Toolchain setup" in Lab 1 with a reproducible container.
# Base is Ubuntu 24.04 to match the course's WSL/Ubuntu target.
FROM ubuntu:24.04

# Pin the packages that are NOT in apt (the reason the lab pushes you to Homebrew).
ARG ONNXRUNTIME_VERSION=1.27.0
ARG AUTODIFF_VERSION=1.1.2
ARG NANOBENCH_VERSION=4.3.11
# OpenCV is built from source: Ubuntu 24.04's libopencv-dev is 4.6.0, but the
# labs target 4.7 (ArUco moved into the main objdetect module in 4.7).
ARG OPENCV_VERSION=4.7.0

# User to match the host so files created in the mounted workspace stay yours.
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

# ---------------------------------------------------------------------------
# 1. Stable build toolchain + library dev headers needed to compile OpenCV/VTK.
#    Kept in its own layer (separate from the volatile dev tools in section 6)
#    so tweaking quality-of-life packages doesn't invalidate the OpenCV build.
#
#    BuildKit cache mounts persist apt's package cache + index between builds:
#    dropping docker-clean lets the downloaded .debs survive, so a rebuild of
#    this layer re-installs from cache instead of re-downloading. The mounts
#    aren't committed to the image, so /var/lib/apt/lists stays out of it (no
#    need for the old `rm -rf /var/lib/apt/lists/*`).
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
        rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update && apt-get install -y --no-install-recommends \
        # --- compiler / build system ---
        build-essential \
        # gcc-14's libstdc++ provides <print> for C++23 (clang picks the
        # newest installed libstdc++ automatically)
        g++-14 \
        clang \
        lld \
        cmake \
        ninja-build \
        pkg-config \
        # ccache caches compiled object files (via the cache mount below), so
        # even a full OpenCV rebuild only recompiles what actually changed.
        ccache \
        # --- vcs / fetch tools ---
        git \
        curl \
        wget \
        ca-certificates \
        unzip \
        # --- numerical / vision libraries ---
        libboost-all-dev \
        libeigen3-dev \
        libsuitesparse-dev \
        libvtk9-dev \
        # --- OpenCV 4.7 build deps (built from source below; apt has only 4.6) ---
        libtbb-dev \
        libgtk-3-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libavutil-dev \
        libwebp-dev \
        # --- VTK external dev deps so find_package(VTK) resolves cleanly ---
        #     (Ubuntu's VTK CMake config references these; see the helper below)
        libglew-dev \
        libfreetype-dev \
        libx11-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libxml2-dev \
        libexpat1-dev \
        libtiff-dev \
        libpng-dev \
        libjpeg-dev \
        libsqlite3-dev \
        libdouble-conversion-dev \
        libjsoncpp-dev \
        zlib1g-dev

# Work around Ubuntu's libvtk9-dev shipping a CMake config that references
# external targets for modules you may not request, which trips find_package(VTK)
# on modern CMake. This helper (an official VTK injection point) finds the real
# render/IO deps and stubs the exotic unused ones. See the file for details.
COPY vtk-find-package-helpers.cmake \
     /usr/lib/x86_64-linux-gnu/cmake/vtk-9.1/vtk-find-package-helpers.cmake

# Make clang the default C/C++ compiler (mirrors "update-alternatives" in the lab).
RUN update-alternatives --install /usr/bin/cc  cc  /usr/bin/clang   100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 100
ENV CC=clang \
    CXX=clang++ \
    CMAKE_GENERATOR=Ninja

# ---------------------------------------------------------------------------
# 2. autodiff (header-only, not in apt) -> installs autodiffConfig.cmake
# ---------------------------------------------------------------------------
RUN git clone --depth 1 --branch "v${AUTODIFF_VERSION}" \
        https://github.com/autodiff/autodiff.git /tmp/autodiff \
    && cmake -S /tmp/autodiff -B /tmp/autodiff/build -G Ninja \
        -DAUTODIFF_BUILD_TESTS=OFF \
        -DAUTODIFF_BUILD_PYTHON=OFF \
        -DAUTODIFF_BUILD_EXAMPLES=OFF \
        -DAUTODIFF_BUILD_DOCS=OFF \
    && cmake --build /tmp/autodiff/build --target install \
    && rm -rf /tmp/autodiff

# ---------------------------------------------------------------------------
# 3. nanobench (single header, not in apt). Install the header plus a tiny
#    CMake package so both #include <nanobench.h> and find_package(nanobench)
#    work. (Its own CMake pulls test deps via CPM, so we skip that.)
# ---------------------------------------------------------------------------
RUN curl -fsSL \
        "https://raw.githubusercontent.com/martinus/nanobench/v${NANOBENCH_VERSION}/src/include/nanobench.h" \
        -o /usr/local/include/nanobench.h \
    && mkdir -p /usr/local/lib/cmake/nanobench \
    && cat > /usr/local/lib/cmake/nanobench/nanobench-config.cmake <<'EOF'
# Minimal package config for the header-only nanobench.
if(NOT TARGET nanobench::nanobench)
    add_library(nanobench::nanobench INTERFACE IMPORTED)
    set_target_properties(nanobench::nanobench PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "/usr/local/include")
endif()
EOF

# ---------------------------------------------------------------------------
# 4. ONNX Runtime (prebuilt CPU binaries, not in apt) -> /usr/local
#    The tarball ships a CMake package config (onnxruntimeConfig.cmake) that was
#    generated for a lib64 + include/onnxruntime layout: its imported target
#    references ${prefix}/lib64/libonnxruntime.so and ${prefix}/include/onnxruntime.
#    The tarball itself keeps libs in lib/ and headers flat in include/, so we
#    install headers into an onnxruntime/ subdir and alias lib64 -> lib to make
#    find_package(onnxruntime) resolve. (Headers also kept flat for any code that
#    includes them directly off the default include path.)
# ---------------------------------------------------------------------------
RUN curl -fsSL \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" \
        -o /tmp/ort.tgz \
    && tar -xzf /tmp/ort.tgz -C /tmp \
    && cp -a "/tmp/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}/include/." /usr/local/include/ \
    && mkdir -p /usr/local/include/onnxruntime \
    && cp -a "/tmp/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}/include/." /usr/local/include/onnxruntime/ \
    && cp -a "/tmp/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}/lib/."     /usr/local/lib/ \
    && ln -sfn lib /usr/local/lib64 \
    && ldconfig \
    && rm -rf /tmp/ort.tgz "/tmp/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}"

# ---------------------------------------------------------------------------
# 5. OpenCV 4.7 (main + contrib) from source. Ubuntu apt only ships 4.6.0, and
#    the labs need 4.7's objdetect-based ArUco plus the contrib aruco.hpp
#    umbrella header. BUILD_LIST is trimmed to the modules the labs use so the
#    build stays reasonable (drops dnn/gapi/stitching/ml/photo/etc).
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.ccache \
        git clone --depth 1 --branch "${OPENCV_VERSION}" \
        https://github.com/opencv/opencv.git /tmp/opencv \
    && git clone --depth 1 --branch "${OPENCV_VERSION}" \
        https://github.com/opencv/opencv_contrib.git /tmp/opencv_contrib \
    && cmake -S /tmp/opencv -B /tmp/opencv/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
        -DBUILD_LIST=core,imgproc,imgcodecs,videoio,highgui,features2d,flann,calib3d,objdetect,video,aruco \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_opencv_python3=OFF \
        -DWITH_TBB=ON \
        -DWITH_GTK=ON \
        -DOPENCV_GENERATE_PKGCONFIG=ON \
    && cmake --build /tmp/opencv/build --target install \
    && ccache --show-stats \
    && ldconfig \
    && rm -rf /tmp/opencv /tmp/opencv_contrib

# ---------------------------------------------------------------------------
# 6. Volatile dev tooling + runtime bits: analysis/format/debug tools, the
#    GUI/OpenGL runtime for cv::imshow + VTK, and misc quality-of-life. These
#    change more often than the build deps above, so keeping them in a late
#    layer means editing this list never re-triggers the OpenCV compile.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
        apt-get update && apt-get install -y --no-install-recommends \
        # --- clang tooling / debugger ---
        clangd \
        clang-format \
        clang-tidy \
        gdb \
        # --- analysis / test / docs ---
        cppcheck \
        doctest-dev \
        doxygen \
        graphviz \
        # --- GUI / OpenGL runtime for cv::imshow and VTK ---
        libgl1-mesa-dri \
        mesa-utils \
        x11-apps \
        # --- misc quality-of-life ---
        sudo \
        locales \
        less \
        nano

# ---------------------------------------------------------------------------
# 7. Non-root user matching the host UID/GID (ubuntu:24.04 ships a uid-1000
#    'ubuntu' user; drop it first so our uid is free).
# ---------------------------------------------------------------------------
RUN userdel -r ubuntu 2>/dev/null || true; \
    groupadd --gid "${USER_GID}" "${USERNAME}" 2>/dev/null || true; \
    useradd --uid "${USER_UID}" --gid "${USER_GID}" -m -s /bin/bash "${USERNAME}"; \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/"${USERNAME}"; \
    chmod 0440 /etc/sudoers.d/"${USERNAME}"

USER ${USERNAME}
WORKDIR /workspace
CMD ["/bin/bash"]
