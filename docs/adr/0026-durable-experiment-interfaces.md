# ADR-0026: Durable experiment interfaces

## Status

Accepted

## Context

An evaluation must retain enough typed configuration and provenance to be
replayed without silently changing its benchmark projection, components, or
completed attempts. The built-in adapters cover common benchmarks, but local
datasets need a bounded way to map nested records into samples. Domain-specific
adapters and graders also need an explicit extension point without ambient
plugin discovery or imports from host systems.

Restarting a partial run has the same requirement. Replaying a completed
attempt changes cost and may repeat a side effect; resuming after a manifest,
resolved dataset, or prepared sample has changed would join incomparable work.

## Decision

Manifests may declare `field_map`, a typed path-only projection from JSON-like
records to an evaluation sample. Paths are literal object keys and nonnegative
list indexes. Expressions, template evaluation, and dynamic imports are not
part of this interface.

For domain components, a manifest may select one local Python source file,
factory function, and JSON-like `config` block. The factory receives that
config and returns one `ComponentBundle` containing a public benchmark adapter
and grader. The CLI records distinct SHA-256 fingerprints for the source,
function name, and configuration, plus their combined identity, in the run
manifest. There is no entry-point scan or implicit component selection.

`evalkit run --checkpoint-dir` uses a locked checkpoint store. It persists
redacted completed attempts, reuses them only with `--resume`, and checks a
canonical identity of the manifest, resolved dataset, and prepared samples.
Any semantic drift requires a new checkpoint directory. Interrupted work is
never replayed unless `--retry-interrupted` is explicit.

## Alternatives

Allowing Python expressions in field maps was rejected because configuration
would become executable code with unclear provenance. Ambient plugin discovery
was rejected because an installed package could change a run without a manifest
change. Replaying all attempts on resume was rejected because it duplicates
side effects and costs. Ignoring changed manifest or dataset identity was
rejected because it produces a report that looks comparable while combining
different evaluation inputs.

## Consequences

Users can configure ordinary local data without authoring an adapter, and can
use trusted local domain code when that mapping is insufficient. A local factory
is executable code and therefore remains an explicit trusted-operator choice.
Its identity becomes part of persisted provenance and checkpoint compatibility.

Checkpoint directories are single-writer resources and are durable artifacts
that operators must retain with their reports when a run may be resumed.
Factories do not expand the package dependency boundary: the package continues
to reach a system under test through `ExecutionTarget` only.

## Validation

Unit tests cover nested field maps, literal path failures, manifest
serialization, factory identity sensitivity to source/function/config changes,
local-dataset mutation detection, required composite gate behavior, finite
weights, and CLI checkpoint option validation. Checkpoint tests cover completed
attempt reuse, interrupted-attempt opt-in, locking, and semantic-drift refusal.
The dependency-boundary contract scans the package for prohibited sibling
imports, and strict type checking covers the production implementation.

## Supersession

Adding expression evaluation, automatic component discovery, another factory
transport, or checkpoint reuse across a changed canonical identity requires a
new ADR that supersedes this one. Changes to the package dependency boundary
remain governed by ADR-0001.
