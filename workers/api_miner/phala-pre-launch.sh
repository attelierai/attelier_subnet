#!/bin/bash
# Pass this measured script with Phala --pre-launch-script. Credentials must
# come from sealed environment values, never from compose or command arguments.
set -eu
umask 077
if test -n "${DSTACK_DOCKER_REGISTRY:-}"; then
    : "${DSTACK_DOCKER_USERNAME:?sealed registry username required}"
    : "${DSTACK_DOCKER_PASSWORD:?sealed registry password required}"
    printf '%s' "$DSTACK_DOCKER_PASSWORD" | docker login \
        --username "$DSTACK_DOCKER_USERNAME" --password-stdin "$DSTACK_DOCKER_REGISTRY"
fi
# State ownership is prepared by the root-only compose `state-init` service on
# a Docker named volume. dstack persists Docker's data directory on its
# encrypted data disk; an ordinary guest path is not a durable state store.
if test -S /var/run/dstack.sock; then
    chgrp 10001 /var/run/dstack.sock
    chmod 0660 /var/run/dstack.sock
fi
