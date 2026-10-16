"""The lever device tier at L <= 4, both cards (#364).

data/measured/lever_tier.json is the record of ONE consolidated device
session: every lever's device file (`test/test_*lever*_device.py`,
`test_kernel_fused_device.py`, `test_kernel_mixed_device.py`) run with
`-m "gpu and <backend> and not slow"` through `devtools/kernel-tier.sh` on
the RTX 3080 (CUDA, sm_86) and the RX 9060 XT (HIP, gfx1200). Its run half
is a transcription of the two sessions' pytest counts (nothing here can
re-run a card); its `spilling` half is DERIVED from
data/measured/lever_ledger.json (#362) and recomputed here, never written
by hand.

All fast-tier (no compiler, no card):

- the run half's shape: both backends, per-file counts that sum to the
  session's, no failure, and a skip only as a named opt-in (timed, #363);
- every lever device file is in the record and every record names a file
  that exists, so a new lever device file fails this pin until it is run;
- every ledger arm is exercised by some file;
- the spilling half equals `derive_spilling(lever_ledger.json)`;
- what the L <= 4 rule left unrun is stated, by name.

`PYTHONPATH=src uv run python test/test_lever_tier.py --write` rewrites the
`spilling` half only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIER = ROOT / "data" / "measured" / "lever_tier.json"
LEDGER = ROOT / "data" / "measured" / "lever_ledger.json"
BACKENDS = {"cuda": "sm_86", "hip": "gfx1200"}
# arms the run exercises that are not ledger arms (J's composition is a
# device-test parametrisation, not a ledger row)
EXTRA_ARMS = {"fused+mixed"}


def lever_device_files() -> list[str]:
    t = ROOT / "test"
    files = sorted(t.glob("test_*lever*_device.py")) + [
        t / "test_kernel_fused_device.py", t / "test_kernel_mixed_device.py"]
    return sorted(str(f.relative_to(ROOT)) for f in files)


def derive_spilling(ledger: dict) -> dict:
    """Per arch, every class where the literal kernel or any arm spills
    (`spill_store > 0`), with each arm read against the literal: relieves
    (arm < literal), worsens (arm > literal), unchanged. Units are the
    ledger's: bytes on sm_86, spilled registers on gfx1200."""
    by: dict[tuple[str, str], list[dict]] = {}
    for r in ledger["rows"]:
        by.setdefault((r["arch"], r["class"]), []).append(r)
    out: dict[str, list[dict]] = {a: [] for a in BACKENDS.values()}
    for (arch, cls), rows in sorted(by.items()):
        lit = rows[0]["literal_readings"]
        if not (lit["spill_store"] or any(
                r["arm_readings"]["spill_store"] for r in rows)):
            continue
        rel, wor, same = {}, {}, []
        for r in sorted(rows, key=lambda r: r["arm"]):
            a, b = r["arm_readings"]["spill_store"], lit["spill_store"]
            if a < b:
                rel[r["arm"]] = a
            elif a > b:
                wor[r["arm"]] = a
            else:
                same.append(r["arm"])
        out[arch].append({
            "class": cls, "target": rows[0]["target"],
            "literal_registers": lit["registers"],
            "literal_spill_store": lit["spill_store"],
            "relieves": rel, "worsens": wor, "unchanged": same})
    return out


DOC = json.loads(TIER.read_text())


def test_the_run_half_has_both_backends_and_every_case_passed() -> None:
    assert set(DOC["backends"]) == set(BACKENDS)
    for be, arch in BACKENDS.items():
        s = DOC["backends"][be]
        assert s["arch"] == arch and s["device"] == "present"
        assert s["exit"] == 0
        assert s["failed"] == 0 and s["errors"] == 0
        # a skip is allowed only as a named opt-in (timed: #363), never silent
        assert s["skipped"] == len(s["opt_in_skips"])
        assert all(k["reason"] for k in s["opt_in_skips"])
        assert s["passed"] + s["skipped"] == s["collected"] > 0
        per = [f["cases"][be] for f in DOC["files"]]
        assert sum(c["run"] for c in per) == s["collected"]
        assert sum(c["skipped"] for c in per) == s["skipped"]
        assert all(c["passed"] + c["skipped"] == c["run"] and c["passed"] > 0
                   for c in per)


def test_every_lever_device_file_is_recorded_and_exists() -> None:
    got = [f["file"] for f in DOC["files"]]
    assert len(got) == len(set(got))
    assert sorted(got) == lever_device_files()


def test_every_ledger_arm_is_exercised_and_nothing_else() -> None:
    ledger = json.loads(LEDGER.read_text())
    run = {a for f in DOC["files"] for a in f["arms"]}
    assert set(ledger["arms"]) <= run
    assert run - set(ledger["arms"]) <= EXTRA_ARMS


def test_the_spilling_half_is_the_ledger_derived() -> None:
    ledger = json.loads(LEDGER.read_text())
    assert DOC["spilling"] == derive_spilling(ledger), (
        "rerun: PYTHONPATH=src uv run python test/test_lever_tier.py --write")
    assert DOC["max_L"] == ledger["max_L"] == 4


def test_what_the_l4_rule_left_unrun_is_named() -> None:
    names = " ".join(n["what"] for n in DOC["not_run"])
    for fam in ("(dd|dd)", "(dp|dp)", "(21|21)"):
        assert fam in names
    assert all(n["why"] for n in DOC["not_run"])


if __name__ == "__main__" and sys.argv[1:] == ["--write"]:
    doc = json.loads(TIER.read_text())
    doc["spilling"] = derive_spilling(json.loads(LEDGER.read_text()))
    TIER.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
