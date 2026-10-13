#!/bin/sh
# Start an oMLX server from the M1 overlay built by scripts/m1_overlay.py,
# using the app's bundled Python/MLX runtime and the normal ~/.omlx settings.
# Stop the app-managed server first (same port). Extra args go to `omlx serve`.
set -eu

APP_RES="/Applications/oMLX.app/Contents/Resources"
OVERLAY="${OMLX_M1_OVERLAY:-$HOME/.omlx/m1-overlay}"
CPYTHON="$APP_RES/Python/cpython-3.11"
MLX_SITE="$APP_RES/Python/framework-mlx-base/lib/python3.11/site-packages"

if [ ! -d "$OVERLAY/omlx" ]; then
    echo "overlay not found at $OVERLAY/omlx; run scripts/m1_overlay.py first" >&2
    exit 1
fi

export PYTHONHOME="$CPYTHON"
export PYTHONDONTWRITEBYTECODE=1
# `python -m` puts the current directory first on sys.path; run from a source
# checkout that would import its omlx (no native kernels, so no ANE prefill)
# instead of the overlay.
export PYTHONSAFEPATH=1
export PYTHONPATH="$OVERLAY:$MLX_SITE"
# Steel FA-256 key tile: 16 measured faster than the default 8 on M1 Max.
export OMLX_FA256_K_BLOCK="${OMLX_FA256_K_BLOCK:-16}"
# Qwen3.5-family prefill chunks of 2048 (upstream floor 4096 on this host) keep
# the ANE prefill banks: +24% prefill at 15k, +10% at 61k on Qwen3.8-27B.
export OMLX_QWEN35_PREFILL_FLOOR="${OMLX_QWEN35_PREFILL_FLOOR:-2048}"

exec "$CPYTHON/bin/python3" -m omlx.cli serve "$@"
