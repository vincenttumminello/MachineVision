#!/usr/bin/env bash
# Rebuild the MCHA4400 dev-container image.
#
# devcontainer.json is pinned to the prebuilt tag `mcha4400:latest`, so opening
# the container never builds. Run this by hand after editing the Dockerfile,
# then choose "Dev Containers: Rebuild Container" once to pick up the new image
# (or just Reopen — VS Code uses whatever `mcha4400:latest` currently points at).
#
# BuildKit reuses the apt/ccache cache mounts and the unchanged upper layers, so
# only what actually changed recompiles. See daemon.json's builder.gc cap, which
# keeps that cache from being evicted between builds.
set -euo pipefail

cd "$(dirname "$0")"

DOCKER_BUILDKIT=1 docker build \
    --tag mcha4400:latest \
    --build-arg USER_UID="$(id -u)" \
    --build-arg USER_GID="$(id -g)" \
    "$@" \
    .

echo
echo "Built mcha4400:latest — Reopen (or Rebuild) the container in VS Code to use it."
