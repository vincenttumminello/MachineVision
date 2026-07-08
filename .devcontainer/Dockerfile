# MCHA4400 toolchain image
# Replaces the manual "Toolchain setup" in Lab 1 with a reproducible container.
# Base is Ubuntu 24.04 to match the course's WSL/Ubuntu target.
FROM ubuntu:24.04

# Pin the packages that are NOT in apt (the reason the lab pushes you to Homebrew).
ARG ONNXRUNTIME_VERSION=1.27.0
ARG AUTODIFF_VERSION=1.1.2
ARG NANOBENCH_VERSION=4.3.11

# User to match the host so files created in the mounted workspace stay yours.
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

# ---------------------------------------------------------------------------
# 1. Everything that IS available via apt (main + universe are enabled on the
#    official ubuntu:24.04 image).
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        # --- compiler / build system ---
        build-essential \
        # gcc-14's libstdc++ provides <print> for C++23 (clang picks the
        # newest installed libstdc++ automatically)
        g++-14 \
        clang \
        clangd \
        clang-format \
        clang-tidy \
        lld \
        gdb \
        cmake \
        ninja-build \
        pkg-config \
        # --- vcs / fetch tools ---
        git \
        curl \
        wget \
        ca-certificates \
        unzip \
        # --- analysis / test / docs ---
        cppcheck \
        doctest-dev \
        doxygen \
        graphviz \
        # --- numerical / vision libraries ---
        libboost-all-dev \
        libeigen3-dev \
        libsuitesparse-dev \
        libopencv-dev \
        libvtk9-dev \
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
        zlib1g-dev \
        # --- GUI / OpenGL runtime for cv::imshow and VTK ---
        libgl1-mesa-dri \
        mesa-utils \
        x11-apps \
        # --- misc quality-of-life ---
        sudo \
        locales \
        less \
        nano \
    && rm -rf /var/lib/apt/lists/*

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
# ---------------------------------------------------------------------------
RUN curl -fsSL \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" \
        -o /tmp/ort.tgz \
    && tar -xzf /tmp/ort.tgz -C /tmp \
    && cp -a "/tmp/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}/include/." /usr/local/include/ \
    && cp -a "/tmp/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}/lib/."     /usr/local/lib/ \
    && ldconfig \
    && rm -rf /tmp/ort.tgz "/tmp/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}"

# ---------------------------------------------------------------------------
# 5. Non-root user matching the host UID/GID (ubuntu:24.04 ships a uid-1000
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
