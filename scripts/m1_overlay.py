#!/usr/bin/env python3
"""Build a patched copy of the installed oMLX.app package for M1 testing.

The app bundle stays untouched (it is code-signed). This copies
``oMLX.app/Contents/Resources/omlx`` to ``~/.omlx/m1-overlay/omlx`` and applies
the M1 changes from this repository on top:

- ``patches/m1_verify_qmv.py`` and ``patches/m1_verify_mma.py`` (2-4 and
  5-16 row speculative-verify kernels for applegpu_g13) and their routing
  hooks in ``patches/qwen35_verify_qmm.py``
- ``utils/hardware.get_apple_gpu_generation``
- ``speculative/dflash_drafter.py``: DFlash2 drafters converted without
  ``num_target_layers`` take the target's layer count
- ``patches/qwen35_verify_sdpa_split.py``: M1 key-split sizing for the
  tensor-op verify attention
- in ``patches/qwen35_fa256_attention.py``: fp16 dispatch-budget calibration
  and the M1 route that runs prefill attention through query-chunked MLX SDPA

Every edit is an exact-match replacement; if the installed app version does
not contain an expected snippet the build aborts instead of guessing. Run
``scripts/m1_serve.sh`` afterwards to start a server from the overlay.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP_PKG = Path("/Applications/oMLX.app/Contents/Resources/omlx")
DEST = Path.home() / ".omlx" / "m1-overlay"

_M1_ROUTE_HELPER = (
    "from . import m1_verify_mma, m1_verify_qmv\n\n\n"
    "def _m1_route(rows, K, N, bits, group_size, dtype):\n"
    '    """M1-class kernel for a verify projection ("m1", "m1_mma") or None."""\n'
    "    if m1_verify_qmv.eligible(rows, K, N, bits, group_size, dtype):\n"
    '        return "m1"\n'
    "    if m1_verify_mma.eligible(rows, K, N, bits, group_size, dtype):\n"
    '        return "m1_mma"\n'
    "    return None\n"
)

_VERIFY_EDITS = [
    (
        "import logging\nimport threading\n",
        "import logging\nimport threading\n\n" + _M1_ROUTE_HELPER,
    ),
    (
        "def sg8_eligible(M: int, K: int, N: int, bits: int, group_size: int, dtype) -> bool:\n"
        "    import mlx.core as mx\n\n"
        "    nsg, ct = _sg8_geometry(int(K), int(N))",
        "def sg8_eligible(M: int, K: int, N: int, bits: int, group_size: int, dtype) -> bool:\n"
        "    import mlx.core as mx\n\n"
        "    if _m1_route(M, K, N, bits, group_size, dtype):\n"
        "        return False\n"
        "    nsg, ct = _sg8_geometry(int(K), int(N))",
    ),
    (
        "    rows = x.shape[0] * x.shape[1]\n"
        "    K = x.shape[2]\n"
        "    first = linears[0]\n"
        "    for linear in linears:",
        "    rows = x.shape[0] * x.shape[1]\n"
        "    K = x.shape[2]\n"
        "    first = linears[0]\n"
        "    if any(\n"
        "        isinstance(linear, nn.QuantizedLinear)\n"
        "        and _m1_route(\n"
        "            rows, K, linear.scales.shape[0], linear.bits, linear.group_size, x.dtype\n"
        "        )\n"
        "        for linear in linears\n"
        "    ):\n"
        "        return False\n"
        "    for linear in linears:",
    ),
    (
        "        route = None\n"
        "        if sg8_eligible(rows, K, N, self.bits, self.group_size, x.dtype):",
        "        route = _m1_route(rows, K, N, self.bits, self.group_size, x.dtype)\n"
        "        if route:\n"
        "            pass\n"
        "        elif sg8_eligible(rows, K, N, self.bits, self.group_size, x.dtype):",
    ),
    (
        "            x2 = x.reshape(rows, K)\n"
        '            if route == "sg8":',
        "            x2 = x.reshape(rows, K)\n"
        '            if route in ("m1", "m1_mma") and self.scales.dtype != x.dtype:\n'
        "                return orig_call(self, x)\n"
        '            if route == "m1":\n'
        "                y = m1_verify_qmv.m1_qmm(\n"
        "                    x2, self.weight, self.scales, self.biases,\n"
        "                    bits=self.bits, group_size=self.group_size,\n"
        "                )\n"
        '            elif route == "m1_mma":\n'
        "                y = m1_verify_mma.m1_mma_qmm(\n"
        "                    x2, self.weight, self.scales, self.biases,\n"
        "                    bits=self.bits, group_size=self.group_size,\n"
        "                )\n"
        '            elif route == "sg8":',
    ),
]

_FA256_EDITS = [
    (
        "        work = _CALIB_HEADS * _CALIB_Q_LEN * _CALIB_KV_LEN\n",
        "        work = _CALIB_HEADS * _CALIB_Q_LEN * _CALIB_KV_LEN\n"
        "        from omlx.utils.hardware import get_apple_gpu_generation\n\n"
        "        dtype = mx.float16 if get_apple_gpu_generation() == 13 else mx.bfloat16\n",
    ),
    (".astype(\n            mx.bfloat16\n        )", ".astype(dtype)", 3),
]


_KERNEL_CALL = """kernel(
{i}    queries,
{i}    keys,
{i}    values,
{i}    scale,
{i}    causal=True,
{i}    q_block=q_block,
{i}    k_block=k_block,
{i}    dispatch_budget=dispatch_budget,
{i})"""

_ATTEND = """    m1_sdpa = _m1_sdpa_enabled()

    def attend(queries, keys, values, scale):
        if m1_sdpa:
            rows = _m1_query_rows(queries.shape[1], queries.shape[2], keys.shape[2])
            if rows:
                return _m1_chunked_sdpa(queries, keys, values, scale, rows)
        return kernel(
            queries,
            keys,
            values,
            scale,
            causal=True,
            q_block=q_block,
            k_block=k_block,
            dispatch_budget=dispatch_budget,
        )

    patched_any = False
"""


def _fa256_m1_edits() -> list:
    """Copy the M1 SDPA constants/helpers from this repo's fa256 patch."""
    source = (REPO / "omlx" / "patches" / "qwen35_fa256_attention.py").read_text()
    consts = source[source.index("# M1-class GPUs (applegpu_g13)") : source.index("_M1_SDPA_MIN_ROWS = 64")]
    consts += "_M1_SDPA_MIN_ROWS = 64\n_M1_SDPA_CACHE_TRIM = 2 * 1024**3\n"
    helpers = source[source.index("def _m1_sdpa_enabled()") : source.index("def _should_route(")]
    lm_call = _KERNEL_CALL.format(i=" " * 20)
    vlm_call = _KERNEL_CALL.format(i=" " * 24)
    return [
        ("_PATCHED = False\n", "_PATCHED = False\n\n" + consts),
        ("def _should_route(", helpers + "def _should_route("),
        ("    patched_any = False\n", _ATTEND),
        ("return " + lm_call, "return attend(queries, keys, values, scale)"),
        ("return " + vlm_call, "return attend(queries, keys, values, scale)"),
    ]


def _dflash_edits() -> list:
    """DFlash2 drafters without num_target_layers: copy the repo's fallback."""
    source = (REPO / "omlx" / "speculative" / "dflash_drafter.py").read_text()
    helpers = source[source.index("def _target_layer_count(") : source.index("def load_dflash_drafter(")]
    return [
        ("def load_dflash_drafter(", helpers + "def load_dflash_drafter("),
        (
            '    model, kind = load_drafter(path, kind="dflash")',
            "    with _default_num_target_layers(_target_layer_count(target_model)):\n"
            '        model, kind = load_drafter(path, kind="dflash")',
        ),
    ]


def _verify_attn_edits() -> list:
    """M1 key-split sizing for the tensor-op verify attention."""
    source = (REPO / "omlx" / "patches" / "qwen35_verify_sdpa_split.py").read_text()
    helpers = source[source.index("@functools.cache\ndef _m1_gpu()") : source.index("def _gqa_causal_sdpa(")]
    return [
        ("def _gqa_causal_sdpa(queries, keys, values, scale):",
         helpers + "def _gqa_causal_sdpa(queries, keys, values, scale):"),
        ("    if kv_len > 16384:\n        chunk = 512\n",
         "    if kv_len > 16384 and _m1_gpu():\n        chunk = _m1_gqa_chunk(kv_len, kv_heads)\n"
         "    elif kv_len > 16384:\n        chunk = 512\n"),
    ]


def _apply(path: Path, edits) -> None:
    text = path.read_text()
    for edit in edits:
        old, new = edit[0], edit[1]
        count = edit[2] if len(edit) > 2 else 1
        found = text.count(old)
        if found != count:
            sys.exit(
                f"{path.name}: expected {count} match(es) of {old.splitlines()[0]!r}, "
                f"found {found}; the installed app version is not supported"
            )
        text = text.replace(old, new)
    path.write_text(text)


def _hardware_helper() -> str:
    source = (REPO / "omlx" / "utils" / "hardware.py").read_text()
    start = source.index("_GPU_GENERATION: Optional[int] = None")
    end = source.index("def get_io_platform_uuid")
    return "\n\n" + source[start:end].rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--app-package", type=Path, default=APP_PKG)
    ap.add_argument("--dest", type=Path, default=DEST)
    args = ap.parse_args()

    if not args.app_package.is_dir():
        sys.exit(f"app package not found: {args.app_package}")
    pkg = args.dest / "omlx"
    staging = args.dest / "omlx.building"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(args.app_package, staging, symlinks=True)

    for module in ("m1_verify_qmv.py", "m1_verify_mma.py"):
        shutil.copy2(REPO / "omlx" / "patches" / module, staging / "patches")
    _apply(staging / "patches" / "qwen35_verify_qmm.py", _VERIFY_EDITS)
    _apply(staging / "patches" / "qwen35_fa256_attention.py", _FA256_EDITS + _fa256_m1_edits())
    _apply(staging / "speculative" / "dflash_drafter.py", _dflash_edits())
    _apply(staging / "patches" / "qwen35_verify_sdpa_split.py", _verify_attn_edits())
    hardware = staging / "utils" / "hardware.py"
    if "def get_apple_gpu_generation" not in hardware.read_text():
        hardware.write_text(hardware.read_text() + _hardware_helper())

    if pkg.exists():
        shutil.rmtree(pkg)
    staging.rename(pkg)
    print(f"overlay ready: {pkg}")


if __name__ == "__main__":
    main()
