"""The pipeline (steps 1-3 composed) and `python -m felizjueves.plan`, in-process.

Host-side counting only: nothing here compiles or launches.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from felizjueves.literature.rules import Status, Verdict
from felizjueves.plan.__main__ import (
    EXIT_COULD_NOT_RUN,
    EXIT_FAIL,
    EXIT_PASS,
    EXIT_UNSET,
    SEARCHED_CUTS,
    exit_status,
    main,
)
from felizjueves.plan.batching import partition
from felizjueves.plan.cost import cost
from felizjueves.plan.dag import batch_dag
from felizjueves.plan.pipeline import plan
from felizjueves.plan.sieve import NO_SIEVE, sieve
from felizjueves.spec import Operator, Spec, Target

DATA = Path(__file__).resolve().parents[1] / "data"


def _spec(L: int, n_batches: int, levers: tuple[str, ...] = ()) -> Spec:
    return Spec(Operator.COULOMB, 0, (L, 0, 0, 0), Target.J, levers, n_batches)


# -- the pipeline --------------------------------------------------------------


@pytest.mark.parametrize("sieved", [True, False], ids=["sieve", "no-sieve"])
@pytest.mark.parametrize("n_batches", [1, 3, 6, 9])
@pytest.mark.parametrize("L", range(7, 17))
def test_plan_matches_the_stages_by_hand(L: int, n_batches: int, sieved: bool):
    dags = tuple(batch_dag(L, b) for b in partition(L, n_batches))
    if sieved:
        dags = tuple(sieve(d) for d in dags)
    levers = () if sieved else (NO_SIEVE,)
    got = plan(_spec(L, n_batches, levers))
    assert got.cost == cost(dags)
    assert [d.batch for d in got.dags] == [d.batch for d in dags]


@pytest.mark.parametrize("levers", [(), (NO_SIEVE,)])
def test_the_batch_report_adds_up_to_the_cost(levers: tuple[str, ...]) -> None:
    p = plan(_spec(12, 6, levers))
    assert len(p.batches) == len(p.dags) == 6
    assert sum(b.flops for b in p.batches) == p.cost.flops
    assert max(b.intermediates for b in p.batches) == p.cost.max_intermediates
    assert max(b.targets for b in p.batches) == p.cost.max_targets
    assert [(b.direction, b.a, b.b) for b in p.batches] == [d.batch for d in p.dags]


def test_the_report_is_plain_json() -> None:
    report = plan(_spec(7, 3)).report()
    assert json.loads(json.dumps(report)) == report
    assert report["spec"] == _spec(7, 3).key()
    assert report["cost"]["flops"] == 425  # [JEngine] Table 1's anchor, exact


def test_an_unknown_lever_is_still_refused() -> None:
    with pytest.raises(ValueError, match="no lever is implemented"):
        plan(_spec(7, 3, ("bogus",)))


# -- exit status ---------------------------------------------------------------


def _v(status: Status) -> Verdict:
    return Verdict("r", "t", status, "")


@pytest.mark.parametrize(("statuses", "code"), [
    ([Status.PASS], EXIT_PASS),
    ([Status.PASS, Status.INAPPLICABLE], EXIT_PASS),
    ([Status.PASS, Status.UNSET], EXIT_UNSET),
    ([Status.FAIL, Status.UNSET], EXIT_FAIL),
    ([Status.INAPPLICABLE], EXIT_UNSET),
    ([], EXIT_UNSET),
])
def test_exit_status(statuses: list[Status], code: int) -> None:
    assert exit_status([_v(s) for s in statuses]) == code


def test_the_exit_codes_are_distinct() -> None:
    assert len({EXIT_PASS, EXIT_FAIL, EXIT_COULD_NOT_RUN, EXIT_UNSET}) == 4


# -- the CLI, in-process -------------------------------------------------------


def test_replay_literal(capsys: pytest.CaptureFixture[str]) -> None:
    # Every Agreement passes at its measured threshold. Table 3's
    # exact_given_inputs passes from the inputs #14 recorded; Tables 1-2's
    # are UNSET (none recorded), so the run says "nothing to judge" rather
    # than pass.
    assert main(["replay", "literal"]) == EXIT_UNSET
    lines = capsys.readouterr().out.splitlines()
    pairs = [tuple(line.split()[1:4]) for line in lines]
    assert pairs == [
        (f"jengine/table{n}", rule, status)
        for n in (1, 2, 3)
        for rule, status in (
            ("agreement", "PASS"),
            ("exact_given_inputs", "PASS" if n == 3 else "UNSET"),
        )
    ]


@pytest.fixture
def data(tmp_path: Path) -> Path:
    """A copy of data/ a test may change."""
    shutil.copytree(DATA, tmp_path / "data")
    return tmp_path / "data"


def test_replay_literal_fails_on_a_raised_floor(
    data: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = data / "scoreboards" / "jengine" / "literal.json"
    raw = json.loads(path.read_text())
    raw["tables"]["jengine/table1"][0]["floor_exact"] = 40
    path.write_text(json.dumps(raw))
    assert main(["replay", "literal", "--data", str(data)]) == EXIT_FAIL
    assert "table1  agreement  FAIL" in capsys.readouterr().out


def test_replay_searched_without_cuts_is_unset(
    data: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (data / "scoreboards" / "jengine" / SEARCHED_CUTS).unlink()  # #13 records it
    assert main(["replay", "searched", "--data", str(data)]) == EXIT_UNSET
    assert "no candidates yet" in capsys.readouterr().out


def test_replay_searched(capsys: pytest.CaptureFixture[str]) -> None:
    # #13's recorded choices, replayed: Table 3 is at or under everywhere,
    # Tables 1 and 2 are not (test_scoreboards.py pins the cells), so the run
    # FAILs -- and prints each cell's margin, indented under its verdict.
    assert main(["replay", "searched"]) == EXIT_FAIL
    lines = capsys.readouterr().out.splitlines()
    verdicts = [line for line in lines if not line.startswith(" ")]
    assert [tuple(line.split()[1:4]) for line in verdicts] == [
        ("jengine/table1", "at_or_under", "FAIL"),
        ("jengine/table2", "at_or_under", "FAIL"),
        ("jengine/table3", "at_or_under", "PASS"),
    ]
    assert '"worst_overshoot": "+2.04%"' in verdicts[0]
    assert len(lines) == 3 + 21 + 3
    assert "    (16, 1)  +2.04%" in lines


def test_replay_searched_with_an_unread_cuts_file_could_not_run(
    data: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (data / "scoreboards" / "jengine" / SEARCHED_CUTS).write_text("{}")
    assert main(["replay", "searched", "--data", str(data)]) == EXIT_COULD_NOT_RUN
    assert "could not run" in capsys.readouterr().err


def test_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["report", "9", "3", "--no-sieve"]) == EXIT_PASS
    report = json.loads(capsys.readouterr().out)
    assert report == plan(_spec(9, 3, (NO_SIEVE,))).report()
    assert report["levers"] == [NO_SIEVE]


def test_report_refuses_a_bad_batch_count(capsys: pytest.CaptureFixture[str]):
    assert main(["report", "9", "4"]) == EXIT_COULD_NOT_RUN
    assert "could not run" in capsys.readouterr().err
