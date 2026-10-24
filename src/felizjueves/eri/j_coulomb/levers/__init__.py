"""Kernel-body levers of the `[p|q]` kernel (`Target.J`), one module each.

`kernel.py` builds the literal kernel and, when `Spec.kernel_body_levers()`
names one of these, hands the piece the lever changes to its module through
a small hook. The vocabulary entry (name, issue, legality) is
`levers.VOCABULARY`'s; the behaviour lives here, so `kernel.py` stays the
literal reconstruction plus one branch per lever.

    fused   #344: contract each [r]^(0) into J_p as the recurrence forms it
"""
