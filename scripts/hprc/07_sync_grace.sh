#!/bin/bash
# scripts/hprc/07_sync_grace.sh
# Sync selected repo artifacts/data between local machine and TAMU Grace.
#
# Examples:
#   bash scripts/hprc/07_sync_grace.sh pull-runs
#   NETID=abc123 bash scripts/hprc/07_sync_grace.sh pull-data
#   NETID=abc123 bash scripts/hprc/07_sync_grace.sh push-configs

set -euo pipefail

ACTION=${1:-}
if [[ -z "$ACTION" ]]; then
  echo "Usage: bash scripts/hprc/07_sync_grace.sh {pull-runs|pull-data|push-configs|push-repo}"
  exit 1
fi

NETID=${NETID:-""}
if [[ -z "$NETID" ]]; then
  echo "Error: NETID is required, e.g. NETID=abc123 bash scripts/hprc/07_sync_grace.sh pull-runs"
  exit 1
fi

GRACE_HOST=${GRACE_HOST:-"${NETID}@grace.hprc.tamu.edu"}
GRACE_REPO=${GRACE_REPO:-"/scratch/user/${NETID}/temp-data-pipeline"}
LOCAL_REPO=$(pwd)
RSYNC_OPTS=(-av --progress)

case "$ACTION" in
  pull-runs)
    mkdir -p "$LOCAL_REPO/runs"
    rsync "${RSYNC_OPTS[@]}" "$GRACE_HOST:$GRACE_REPO/runs/" "$LOCAL_REPO/runs/"
    ;;
  pull-data)
    mkdir -p "$LOCAL_REPO/data"
    rsync "${RSYNC_OPTS[@]}" \
      --include="clean/***" \
      --include="train/***" \
      --include="raw/forecasts/***" \
      --include="raw/era5/***" \
      --exclude="*" \
      "$GRACE_HOST:$GRACE_REPO/data/" "$LOCAL_REPO/data/"
    ;;
  push-configs)
    rsync "${RSYNC_OPTS[@]}" \
      --include="configs/***" \
      --include="scripts/hprc/***" \
      --include="docs/HPRC_SETUP.md" \
      --exclude="*" \
      "$LOCAL_REPO/" "$GRACE_HOST:$GRACE_REPO/"
    ;;
  push-repo)
    rsync "${RSYNC_OPTS[@]}" \
      --exclude=".git/" \
      --exclude=".venv*/" \
      --exclude="__pycache__/" \
      --exclude="*.pyc" \
      --exclude="data/" \
      --exclude="runs/" \
      "$LOCAL_REPO/" "$GRACE_HOST:$GRACE_REPO/"
    ;;
  *)
    echo "Unknown action: $ACTION"
    echo "Expected one of: pull-runs, pull-data, push-configs, push-repo"
    exit 1
    ;;
esac
