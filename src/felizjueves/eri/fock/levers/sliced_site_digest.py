"""The interleaved digest per slice (#359, milestone 13 lane H3).

#356's sliced HRR runs each bra pair's slice as

    { the slice's HRR, every node     the hrr hook on the slice's context
      the tap of the slice's ERIs     (eris=True)
      the predigest of its ERIs }     six multiply-adds each

so every ERI of the slice is live from its production until the predigest
after the slice's last node (`sliced_hrr.slice_peaks` counts them live to the
end). This lever digests each ERI AT ITS PRODUCTION SITE instead (the
predecessor's ledger row 1, applied per slice): right after the statement
that declares an ERI's HRR node come its tap write and its six multiply-adds,
so the ERI dies there.

    { for each statement of the slice's HRR:
        the statement
        if it produced ERI t_i, and t_0 .. t_i are all produced:
          the tap of t_i; the predigest of t_i }

ORDER-PRESERVING, SO BIT FOR BIT. Each buffer element is a running sum over
the slice's ERIs, so its value depends on the ORDER the ERIs are added in. An
ERI is digested at the first point where it AND every ERI before it (in
`hgp.hrr.targets` order, the predigest's) are produced, which is its
production site whenever the HRR produces the ERIs in that order. In this
tree it always does: within a slice the ERIs are its last `n_c n_d` nodes, in
targets order (`hgp.hrr.order_key` puts the full `(a b | c d)` level last,
ties by Cartesian order), which the test pins for every class. So every
buffer receives the same multiply-adds in the same order as under #356 and
the literal kernel, and `F` and the tap `g` are theirs exactly (test/
test_f_lever_sliced_site_digest.py, interpreted, every class with L <= 4). An
`hrr` hook whose order produced the ERIs otherwise (#361's sub-slices) would
still be exact: the rule only delays a digest, never reorders one.

THE EXPECTED OUTCOME IS A NULL, AND THE LIVE MODEL SAYS SO. Because the ERIs
are the slice's last nodes, interleaving shortens only their own tails: in
`hgp.vrr.peak_live`'s model a slice's live set with the ERIs dying at once
(`interleaved_peaks`) is #356's (`sliced_hrr.slice_peaks`) at every class but
`(ss|pp)`, where it is 10 against 11, and the phase never moves #350's group
count (`n_groups`, the rule at the interleaved floor, equals #356's at all 21
classes; pinned). The predecessor measured row 1 alone as NO GAIN; the
compile figures are recorded either way (the PR, and #362's ledger).

A COMPOSITION ON #356, NOT A STANDALONE LEVER: `sliced-site-digest` REQUIRES
`sliced-hrr` (and so `batched-vrr`) in the same spec
(`batched-vrr+sliced-hrr+sliced-site-digest`). Without the slices the HRR is
one phase whose ERIs are its last level too, so the digest would move inside
the whole HRR, a different lever (row 1 alone, the predecessor's null), which
no key segment here names. The name says "site" (the digest at each ERI's
production site), not "interleave": `interleave` is #343's K lever and the
K tests reserve every lever name containing "interleave" or "split" for it
and its composition. The name sorts after `sliced-hrr` on purpose:
`hooks_for` applies levers in key order, so `apply` runs on #356's hooks and
checks that it does (`contracted` is #356's own), refusing an earlier or a
foreign `contracted` (a lever that supplied its own slice body would have its
body dropped or would drop this one: refused, never merged silently).

WITH #360's `slim-fock` (in flight, PR #395, not on this tree's main) THE
PAIR IS REFUSED, not composed: the key sorts `sliced-site-digest` before
`slim-fock`, so this lever installs its `contracted` first and #360's
`apply`, which accepts only #356's own `contracted`, refuses it as "another
lever's, which this lever would replace silently". That is the honest state:
nothing here names #360. The composition is a follow-up once #395 lands --
`slim_fock.contracted_over(body)` with this module's `body` -- and would be
order-preserving per buffer too (#360's re-association is per atomic, and
this lever only moves where each ERI's statements sit inside a slice).

THE HOOKS IT MOVES: `contracted` (#356's, with this body,
`sliced_hrr.contracted_with(body)`) and `nest` (#350's at `n_groups`, the
same count as #356's). `hrr`, `eri`, `predigest_eri`, `digest_setup`,
`zero_buffers` and `flush_ket` are called through the composed hooks, as
#356 calls them, so a lever wrapping one of them composes.

Like the seams, nothing here imports `plan/`, `ao/`, the interpreter,
`reference/`, a backend, a device or `kernel_f`.
"""

from __future__ import annotations

import dataclasses
from functools import cache

from felizjueves.eri.fock.levers import batched_vrr as _bv
from felizjueves.eri.fock.levers import sliced_hrr as _sh
from felizjueves.eri.fock.levers.seams import OUT, FContext, FHooks, cart_names
from felizjueves.hgp import hrr as _hrr
from felizjueves.hgp.recurrence import hrr_name
from felizjueves.hgp.vrr import AmClass, peak_live
from felizjueves.ir import Assign, Decl, Index, Lit, Stmt
from felizjueves.spec import Spec

__all__ = [
    "NAME",
    "REFUSAL",
    "REQUIRES",
    "apply",
    "body",
    "contracted",
    "digest_points",
    "interleaved_peaks",
    "interleaved_phase",
    "n_groups",
]

NAME = "sliced-site-digest"
REQUIRES = _sh.NAME

REFUSAL = (
    "sliced-site-digest requires sliced-hrr (and so batched-vrr) in the same "
    "spec (#359): it digests each ERI at its production site INSIDE a bra-"
    "pair slice; without the slices the digest would move inside the whole "
    "HRR, row 1 alone, a lever no key segment names")


# ---------------------------------------------------------------------------
# The declared functions of the class
# ---------------------------------------------------------------------------


@cache
def interleaved_peaks(cls: AmClass) -> tuple[int, ...]:
    """Each slice's live set with its ERIs dying at their production (the
    digest consumes each at once): `peak_live` with nothing live out."""
    return tuple(peak_live(s.order, lambda n, s=s: (t.child
                                                    for t in s.terms[n]))
                 for s in _hrr.slices(tuple(cls)))


def interleaved_phase(cls: AmClass) -> int:
    """The interleaved HRR phase's demand: the largest slice's live set."""
    return max(interleaved_peaks(tuple(cls)))


def n_groups(cls: AmClass) -> int:
    """#350's group count at the interleaved floor."""
    return _bv.n_groups(tuple(cls), floor=interleaved_phase(cls))


# ---------------------------------------------------------------------------
# The body
# ---------------------------------------------------------------------------


def digest_points(stmts: tuple[Stmt, ...], targets) -> tuple[int, ...]:
    """Per target (in order), the index of the statement after which it is
    digested: the first point where it and every target before it are
    declared. Refused when a target is not declared exactly once."""
    at: dict[str, list[int]] = {}
    for i, s in enumerate(stmts):
        if isinstance(s, Decl):
            at.setdefault(s.name, []).append(i)
    out, done = [], -1
    for t in targets:
        where = at.get(hrr_name(t), [])
        if len(where) != 1:
            raise ValueError(f"{NAME}: the slice's HRR declares ERI "
                             f"{hrr_name(t)} {len(where)} times, not once")
        done = max(done, where[0])
        out.append(done)
    return tuple(out)


def body(sub: FContext) -> tuple[Stmt, ...]:
    """One slice: its HRR (the `hrr` hook) with each ERI's tap write and
    predigest placed right after it is produced (module docstring)."""
    sl: _sh.Slice = sub.extra[_sh.NAME]
    stmts = tuple(sub.hooks.hrr(sub))
    comps = cart_names(sub.cls)
    after: dict[int, list[Stmt]] = {}
    for i, t, p in zip(sl.indices, sl.dag.targets,
                       digest_points(stmts, sl.dag.targets), strict=True):
        put = after.setdefault(p, [])
        if sub.eris:
            put.append(Assign(Index(OUT, (sub.bra, sub.k, Lit(i))),
                              sub.hooks.eri(sub, t)))
        put += sub.hooks.predigest_eri(sub, t, comps[i])
    out: list[Stmt] = []
    for i, s in enumerate(stmts):
        out += (s, *after.get(i, ()))
    return tuple(out)


contracted = _sh.contracted_with(body)


# ---------------------------------------------------------------------------
# The lever
# ---------------------------------------------------------------------------


def apply(hooks: FHooks, spec: Spec) -> FHooks:
    """#356's slices with the interleaved body. Refused without
    `sliced-hrr` (`REFUSAL`), or when `contracted` is not #356's own."""
    if REQUIRES not in spec.kernel_body_levers():
        raise ValueError(f"{spec.key()}: {REFUSAL}")
    if hooks.contracted is not _sh.contracted:
        raise ValueError(
            f"{spec.key()}: {NAME} replaces sliced-hrr's slice body, but "
            "`contracted` is not sliced-hrr's own (applied before it, or "
            "moved by another lever whose change would be dropped silently)")
    return dataclasses.replace(hooks, contracted=contracted,
                               nest=_bv.nest_at(n_groups(tuple(spec.am))))
