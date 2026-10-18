"""Record vectors as the SoA kernels take them (#421).

    axes("PA", vecs) -> {"XPA": ..., "YPA": ..., "ZPA": ...}

A kernel that carries a `PrimPairSoA` (`prim_pair.py`) reads every record
vector as three flat per-axis arrays, never one interleaved `[n, 3]`; the
interpreted tests build their inputs through this one splitter.
"""

from __future__ import annotations

import numpy as np

from felizjueves.prim_pair import AXES


def axes(name: str, vecs) -> dict[str, np.ndarray]:
    """`vecs` (`[n, 3]`, any sequence) as `X<name>`, `Y<name>`, `Z<name>`,
    each a contiguous `float64[n]`."""
    v = np.asarray(vecs, dtype=np.float64).reshape(-1, 3)
    return {f"{a}{name}": v[:, d].copy() for d, a in enumerate(AXES)}


def joined(args: dict, name: str) -> np.ndarray:
    """The inverse: `X<name> Y<name> Z<name>` of `args` as one `[n, 3]`."""
    return np.stack([np.asarray(args[f"{a}{name}"]) for a in AXES], axis=1)
