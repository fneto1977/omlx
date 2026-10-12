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
export PYTHONPATH="$OVERLAY:$MLX_SITE"
# Steel FA-256 key tile: 16 measured faster than the default 8 on M1 Max.
export OMLX_FA256_K_BLOCK="${OMLX_FA256_K_BLOCK:-16}"

exec "$CPYTHON/bin/python3" -m omlx.cli serve "$@"
