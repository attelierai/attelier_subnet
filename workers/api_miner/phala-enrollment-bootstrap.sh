#!/bin/sh
# Optional one-shot enrollment export for production CVMs without SSH/log access.
# This sends evidence for review; a successful upload grants no admission.
set -eu
umask 077
: "${ATTELIER_MINER_STATE_DIR:?state directory required}"
: "${ATTELIER_ENROLLMENT_CHALLENGE:?operator challenge required}"
: "${ATTELIER_ENROLLMENT_TOKEN:?sealed enrollment token required}"
: "${ATTELIER_WORKER_ID:?worker identity required}"
: "${ATTELIER_COORDINATOR_URL:?reviewed HTTPS gateway required}"
case "$ATTELIER_COORDINATOR_URL" in https://*) ;; *) exit 1 ;; esac
marker="$ATTELIER_MINER_STATE_DIR/enrollment-challenge"
if test ! -f "$marker" || test "$(cat "$marker")" != "$ATTELIER_ENROLLMENT_CHALLENGE"; then
    attelier-api-miner evidence > /tmp/enrollment.json
    printf 'Authorization: Bearer %s\n' "$ATTELIER_ENROLLMENT_TOKEN" > /tmp/enrollment.headers
    curl --fail --silent --show-error --max-time 30 \
        --header @/tmp/enrollment.headers --header 'Content-Type: application/json' \
        --data-binary @/tmp/enrollment.json \
        "${ATTELIER_COORDINATOR_URL%/}/enrollment/$ATTELIER_WORKER_ID" \
        > /tmp/enrollment-receipt.json
    printf '%s\n' "$ATTELIER_ENROLLMENT_CHALLENGE" > "$marker"
fi
exec attelier-api-miner run
