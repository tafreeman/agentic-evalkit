"""Paired target-treatment comparison for arms of one frozen experiment."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from agentic_evalkit.errors import IncompatibleRuns
from agentic_evalkit.models import ExecutionStatus, GradeStatus

if TYPE_CHECKING:
    from agentic_evalkit.models import EvalRunResult


def compare_experiment_arms(
    left: EvalRunResult, right: EvalRunResult, *, names: tuple[str, str]
) -> dict[str, object]:
    """Only the target identity and display name may differ; all other inputs must match.

    Repeated attempts are aggregated within each instance using all-attempts
    success, so correlated attempts are never counted as independent pairs.
    """
    for field in ("environment_fingerprint", "code_fingerprint", "target_fingerprint"):
        if getattr(left.manifest, field) is None or getattr(right.manifest, field) is None:
            raise IncompatibleRuns(message=f"missing {field} provenance")
    excluded = {"run_name", "target_fingerprint"}
    if left.manifest.model_dump(exclude=excluded) != right.manifest.model_dump(exclude=excluded):
        raise IncompatibleRuns(
            message="arms differ in dataset, sampling, grader, or other non-target configuration"
        )
    if left.resolved_dataset != right.resolved_dataset:
        raise IncompatibleRuns(message="arms used different resolved datasets")
    indexes = []
    for run in (left, right):
        index = {(row.sample.sample_id, row.execution.attempt): row for row in run.samples}
        if len(index) != len(run.samples):
            raise IncompatibleRuns(message="duplicate sample-attempt pairs")
        indexes.append(index)
    a, b = indexes
    if set(a) != set(b):
        raise IncompatibleRuns(message="arms do not contain the same sample-attempt identities")
    identifiers = sorted({key[0] for key in a} | {key[0] for key in b})
    outcomes: list[tuple[bool, bool]] = []
    excluded_instances = 0
    for identifier in identifiers:
        pairs = [
            (a.get((identifier, attempt)), b.get((identifier, attempt)))
            for attempt in range(1, left.manifest.attempts + 1)
        ]
        valid = True
        passes_a, passes_b = [], []
        for row_a, row_b in pairs:
            if row_a is None or row_b is None:
                valid = False
                continue
            if row_a.sample != row_b.sample:
                raise IncompatibleRuns(message=f"sample content differs for {identifier}")
            for row in (row_a, row_b):
                if (
                    row.execution.status is not ExecutionStatus.COMPLETED
                    or row.grade is None
                    or row.grade.status not in (GradeStatus.PASS, GradeStatus.FAIL)
                ):
                    valid = False
            passes_a.append(row_a.grade is not None and row_a.grade.status is GradeStatus.PASS)
            passes_b.append(row_b.grade is not None and row_b.grade.status is GradeStatus.PASS)
        if valid:
            outcomes.append((all(passes_a), all(passes_b)))
        else:
            excluded_instances += 1
    n = len(outcomes)
    if n == 0:
        raise IncompatibleRuns(message="no paired instances with definitive grades in both arms")
    a_only = sum(a_pass and not b_pass for a_pass, b_pass in outcomes)
    b_only = sum(b_pass and not a_pass for a_pass, b_pass in outcomes)
    discordant = a_only + b_only
    p_value = min(
        1.0,
        2 * sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1)) / 2**discordant,
    )
    deltas = [int(b_pass) - int(a_pass) for a_pass, b_pass in outcomes]
    rng = random.Random(20260907)  # noqa: S311 - reproducible statistical resampling
    boot = sorted(sum(rng.choices(deltas, k=n)) / n for _ in range(2000))
    return {
        "arms": names,
        "treatment": "target",
        "metric": "all-attempts success per instance",
        "paired_instances": n,
        "excluded_instances": excluded_instances,
        "attempts_per_instance": left.manifest.attempts,
        "left_pass_rate": sum(a for a, _ in outcomes) / n,
        "right_pass_rate": sum(b for _, b in outcomes) / n,
        "delta_right_minus_left": sum(deltas) / n,
        "bootstrap_95_interval": [boot[49], boot[1949]],
        "left_only": a_only,
        "right_only": b_only,
        "mcnemar_exact_p": p_value,
        "qualification": (
            "conditional on definitive paired outcomes; public datasets do not establish "
            "uncontaminated capability"
        ),
    }
