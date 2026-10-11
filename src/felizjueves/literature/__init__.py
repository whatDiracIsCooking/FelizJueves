"""Published numbers, held as data with provenance, and the rules that score
a candidate against them.

Altitude, lowest first: `tables` (a transcribed table, nothing more), `rules`
(a judgement of a candidate against one table), `scoreboard` (which rules,
with which floors, apply to which tables), and `limits` (what a model will
structurally not predict, recorded rather than tested). Each takes what it
needs as an argument -- a path, a table, a candidate -- so none of them knows
where the repo keeps its data or what produced a candidate.
"""
