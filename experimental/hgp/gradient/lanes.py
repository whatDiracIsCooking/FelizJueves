"""The HGP recurrence kernels, run over many primitive quartets at once.

    compile_kernel(kernel)     -> run(scalars, arrays) -> out, one row per lane
    boys_lanes(top, T)         -> [F_0(T), ..., F_top(T)], each an array

``eri.hgp_drive`` runs ``hgp.recurrence``'s VRR kernel in ``ir.interp`` once
per primitive quartet. That is right for a pin, but a gradient sweep needs
eight blocks of a class one higher per shell quartet, which at ~0.15 s per
interpreted ``(pp|pp)`` makes a 6-31G water a matter of hours. So this module
evaluates THE SAME KERNEL, the same ``ir.Kernel`` the builder made, with every
scalar a numpy array over primitive quartets (one lane per quartet), the way a
GPU runs one lane per thread. Nothing about the recurrence is restated here:
the kernel is translated statement by statement into Python source once per
class and ``exec``-ed. Only the statement kinds ``hgp.recurrence``'s two
standalone kernels hold are accepted (``Note``, ``Decl``, ``Assign``, the
prelude's ``Boys``) and only the expression kinds they use (``Lit``, ``Var``,
``Index`` at a literal, ``Call`` of ``ADD SUB MUL DIV SQRT EXP``,
``Weighted``); anything else is refused, as ``hgp_drive.absolute_kernel``
refuses, so a kernel this does not understand cannot be silently misread.

Each ``Weighted`` is spelled ``((c * g) * src)`` summed left to right from 0,
the interpreter's order (``ir.interp``, ``Weighted`` in ``eval``), so a lane
is the interpreter's value up to ``exp`` / ``sqrt`` and the Boys evaluation.
``check.py`` measures the remaining distance against ``hgp_drive.eri_block``.

THE BOYS FUNCTION is ``reference.boys``, vectorised: the same 112-node
Gauss-Legendre rule (``reference.boys._rule``) below ``T_SWITCH``, the same
asymptote from it to ``T_MAX``, refused beyond. ``check.py`` holds it to the
scalar ``reference.boys`` lane by lane.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import cache

import numpy as np

from felizjueves.ir import (
    Assign,
    Call,
    Decl,
    Fn,
    Index,
    Kernel,
    Lit,
    Note,
    Var,
    Weighted,
)
from felizjueves.ir import Boys as BoysFill
from felizjueves.reference import boys as rb

__all__ = ["boys_lanes", "compile_kernel"]

_BINARY = {Fn.ADD: "+", Fn.SUB: "-", Fn.MUL: "*", Fn.DIV: "/"}


def boys_lanes(top: int, T: np.ndarray) -> list[np.ndarray]:
    """``F_m(T)`` for ``m = 0..top`` on every lane: ``reference.boys``'s
    quadrature below ``T_SWITCH`` and its asymptote from there to ``T_MAX``
    (module docstring)."""
    T = np.asarray(T, dtype=float)
    if top > rb.M_MAX:
        raise ValueError(f"m = {top} outside the measured range 0..{rb.M_MAX}")
    if T.size and (T.min() < 0.0 or T.max() > rb.T_MAX):
        raise ValueError(f"T outside the measured range [0, {rb.T_MAX}]")
    t, w = rb._rule(rb.NODES)
    near = T < rb.T_SWITCH
    out = [np.empty_like(T) for _ in range(top + 1)]
    if near.any():
        e = np.exp(-np.outer(T[near], t * t))  # [lanes, nodes]
        for m in range(top + 1):
            out[m][near] = e @ (w * t ** (2 * m))
    far = ~near
    if far.any():
        Tf = T[far]
        a = 0.5 * np.sqrt(math.pi / Tf)
        out[0][far] = a
        for m in range(1, top + 1):
            a = a * ((2 * m - 1) / (2.0 * Tf))
            out[m][far] = a
    return out


def _expr(e) -> str:
    if isinstance(e, Lit):
        return repr(float(e.value)) if isinstance(e.value, float) else repr(e.value)
    if isinstance(e, Var):
        return e.name
    if isinstance(e, Index):
        if len(e.at) != 1 or not isinstance(e.at[0], Lit):
            raise ValueError(f"lanes: index {e!r} is not one literal")
        return f"{e.array}[{e.at[0].value}]"
    if isinstance(e, Call):
        if e.fn in _BINARY:
            a, b = e.args
            return f"({_expr(a)} {_BINARY[e.fn]} {_expr(b)})"
        if e.fn is Fn.SQRT:
            return f"_sqrt({_expr(e.args[0])})"
        if e.fn is Fn.EXP:
            return f"_exp({_expr(e.args[0])})"
        raise ValueError(f"lanes: Call of {e.fn.name} is not a recurrence op")
    if isinstance(e, Weighted):
        acc = "0.0"
        for t in e.terms:
            g = t.g.name_in_kernel
            cg = repr(float(t.c)) if g is None else f"({float(t.c)!r} * {g})"
            acc = f"({acc} + ({cg} * {_expr(t.src)}))"
        return acc
    raise ValueError(f"lanes: {type(e).__name__} is not a recurrence expression")


@cache
def compile_kernel(kernel: Kernel) -> Callable[..., np.ndarray]:
    """``run(scalars, arrays)``: ``kernel`` over lanes. ``scalars`` maps each
    scalar parameter to an array (or a float), ``arrays`` each IN array
    parameter to a sequence of them; the result is ``out`` as an array of
    rows, row ``i`` the kernel's ``out[i]`` on every lane."""
    lines = ["def run(_s, _a):"]
    for p in kernel.params:
        if p.name == "out":
            continue
        src = "_a" if p.array else "_s"
        lines.append(f"    {p.name} = {src}[{p.name!r}]")
    n_out = 1 + max((int(st.dst.at[0].value) for st in kernel.body
                     if isinstance(st, Assign)), default=-1)
    if not n_out:
        raise ValueError(f"lanes: {kernel.name} writes no out")
    lines.append(f"    out = [None] * {n_out}")
    for st in kernel.body:
        if isinstance(st, Note):
            continue
        if isinstance(st, BoysFill):
            lines.append(f"    {st.dst} = _boys({st.top}, {_expr(st.t)})")
        elif isinstance(st, Decl):
            if st.init is None:
                if st.name not in {b.dst for b in kernel.body
                                   if isinstance(b, BoysFill)}:
                    raise ValueError(f"lanes: {st.name} declared, never filled")
                continue
            if st.shape:
                raise ValueError(f"lanes: array Decl {st.name} with an init")
            lines.append(f"    {st.name} = {_expr(st.init)}")
        elif isinstance(st, Assign):
            if not (isinstance(st.dst, Index) and st.dst.array == "out"):
                raise ValueError(f"lanes: assignment to {st.dst!r}")
            lines.append(f"    {_expr(st.dst)} = {_expr(st.value)}")
        else:
            raise ValueError(f"lanes: {type(st).__name__} in {kernel.name} "
                             f"is not a recurrence statement")
    lines.append("    return out")
    ns = {"_boys": boys_lanes, "_sqrt": np.sqrt, "_exp": np.exp}
    exec(compile("\n".join(lines), f"<lanes {kernel.name}>", "exec"), ns)
    return ns["run"]
