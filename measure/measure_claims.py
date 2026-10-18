"""The decision the claims and parity docs cite, read off the tracked store.

test/test_claims.py holds the wording (#166: `jgen` parity has one
difference, Table 4, declined, and the plan is a decision, not a gap); the
grounds those sentences cite -- the tuned plan may not become the default and
the sieve may not leave it, by the rules declared before their rows -- are a
reading of data/measured/kernels.json, so their check is a check of a
measurement artefact and lives here (`devtools/measure.sh check`). Moved from
test_claims.py's `test_the_jgen_difference_is_one_item_and_the_plan_a_decision`.
"""

from __future__ import annotations

from felizjueves import measured_store


def test_the_plan_decision_the_docs_cite_is_the_stores() -> None:
    store = measured_store.load()
    assert measured_store.tuned_may_become_the_default(store) is False
    assert measured_store.sieve_may_leave_the_default(store) is False
