# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Regenerate a vendor backend's op surface from the CUDA backend.

Every ``tex.*`` passthrough op has the same shape across vendors: normalize any
``DType``/quantizer argument, then forward to the vendor extension. The CUDA
backend is kept strictly aligned with TransformerEngine main, so it is the
template; this script copies its op methods into a vendor file, preserving the
parts that are genuinely vendor-specific (library loading, ``_get_tex``,
attention-class wiring, and any op the vendor deliberately overrides).

Usage:
    python tools/gen_vendor_ops.py metax [--check]

``--check`` reports whether the file is already up to date instead of writing,
so CI can fail on drift.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
VENDOR_DIR = REPO / "transformer_engine_plugin_fl" / "backends" / "vendor"
CUDA_FILE = VENDOR_DIR / "cuda" / "cuda.py"

# Markers delimiting the generated region in each vendor file.
BEGIN = "    ##### BEGIN GENERATED OPS (tools/gen_vendor_ops.py) #####"
END = "    ##### END GENERATED OPS #####"

# Methods that are never generated: they are the vendor's own plumbing.
HAND_WRITTEN = {
    "check_available",
    "is_available",
    "get_attention_backend",
    "get_flash_attention_class",
}

# Backends that implement a deliberate subset of the op surface and are
# maintained by hand, like the flagos and reference backends. They fall through
# the manager for anything they do not provide, so there is nothing to generate.
SUBSET_BACKENDS = {"kunlunxin", "npu"}


def _methods(path: pathlib.Path) -> dict[str, str]:
    src = path.read_text()
    lines = src.splitlines()
    tree = ast.parse(src)
    cls = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name.endswith("Backend")
    )
    out = {}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            out[node.name] = "\n".join(lines[start : node.end_lineno])
    return out


def _overridden(path: pathlib.Path) -> set[str]:
    """Ops the vendor file defines outside the generated region."""
    text = path.read_text()
    if BEGIN in text:
        head, _, rest = text.partition(BEGIN)
        _, _, tail = rest.partition(END)
        text = head + tail
    return {m.group(1) for m in re.finditer(r"^    def (\w+)\(", text, re.M)}


def _render(ops: dict[str, str], skip: set[str]) -> str:
    names = [n for n in ops if not n.startswith("_") and n not in skip]
    body = "\n\n".join(ops[n] for n in names)
    return f"{BEGIN}\n{body}\n{END}"


def generate(vendor: str, check: bool) -> int:
    vendor_file = VENDOR_DIR / vendor / f"{vendor}.py"
    if not vendor_file.exists():
        print(f"no such vendor backend: {vendor_file}", file=sys.stderr)
        return 2

    cuda_ops = _methods(CUDA_FILE)
    skip = HAND_WRITTEN | _overridden(vendor_file)
    generated = _render(cuda_ops, skip)

    text = vendor_file.read_text()
    if BEGIN not in text:
        print(
            f"{vendor_file} has no generated region; add the markers first:\n{BEGIN}\n{END}",
            file=sys.stderr,
        )
        return 2

    head, _, rest = text.partition(BEGIN)
    _, _, tail = rest.partition(END)
    new_text = head + generated + tail

    n_ops = generated.count("\n    def ")
    if new_text == text:
        print(f"{vendor}: up to date ({n_ops} generated ops)")
        return 0
    if check:
        print(f"{vendor}: OUT OF DATE -- run tools/gen_vendor_ops.py {vendor}", file=sys.stderr)
        return 1
    vendor_file.write_text(new_text)
    print(f"{vendor}: wrote {n_ops} generated ops, kept {len(skip)} hand-written")
    return 0


def main() -> int:
    vendors = sorted(p.name for p in VENDOR_DIR.iterdir() if (p / f"{p.name}.py").exists())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vendor", nargs="*", help=f"one of: {', '.join(vendors)}")
    parser.add_argument("--check", action="store_true", help="report drift without writing")
    args = parser.parse_args()

    targets = args.vendor or [
        v for v in vendors if v != "cuda" and v not in SUBSET_BACKENDS
    ]
    return max(generate(v, args.check) for v in targets)


if __name__ == "__main__":
    raise SystemExit(main())
