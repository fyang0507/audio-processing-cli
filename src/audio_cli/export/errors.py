"""Typed failures from deterministic transcript export.

The command layer turns these into the repository's fixed-shape refusal payloads.  Keeping
the failures typed here lets the export core stay independent of argparse and stderr.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class ExportError(ValueError):
    """A saved result cannot be exported safely as requested."""


class InvalidResultError(ExportError):
    """A JSON input is not a conforming, exportable result document."""

    def __init__(self, input_path: Path, reason: str) -> None:
        self.input_path = Path(input_path)
        self.reason = reason
        super().__init__(f"{self.input_path}: {reason}")


class IncompatibleResultsError(ExportError):
    """Several valid results cannot be concatenated without guessing."""

    def __init__(
        self,
        input_paths: Sequence[Path],
        reason: str,
        *,
        fix: str | None = None,
    ) -> None:
        self.input_paths = tuple(Path(path) for path in input_paths)
        self.reason = reason
        self.fix = fix
        joined = ", ".join(str(path) for path in self.input_paths)
        super().__init__(f"incompatible result documents ({joined}): {reason}")


class TimingRequiredError(ExportError):
    """A subtitle request has no real word stream from which to take bounds."""

    def __init__(
        self,
        input_path: Path,
        *,
        found: Sequence[str],
        source_path: Path,
        stack: str,
        wants: Sequence[str],
        plan: Mapping[str, Any],
        word_timing_outcome: str | None,
    ) -> None:
        self.input_path = Path(input_path)
        self.found = tuple(found)
        self.source_path = Path(source_path)
        self.stack = stack
        self.wants = tuple(wants)
        self.plan = dict(plan)
        self.word_timing_outcome = word_timing_outcome
        super().__init__(f"{self.input_path} has no produced word timing for subtitle export")


class ReadableTimingRequiredError(ExportError):
    """A readable timestamp request contains a segment without real bounds."""

    def __init__(
        self, input_path: Path, segment_id: str, *, word_timing_outcome: str | None = None
    ) -> None:
        self.word_timing_outcome = word_timing_outcome
        self.input_path = Path(input_path)
        self.segment_id = segment_id
        super().__init__(f"{self.input_path}: {segment_id} has no segment or word timing")


class TimestampsUnsupportedError(ExportError):
    """The readable timestamp option was supplied for another format."""

    def __init__(self, output_format: str) -> None:
        self.output_format = output_format
        super().__init__(f"--timestamps requires txt or md, got {output_format!r}")


class ProvenanceUnsupportedError(ExportError):
    """Provenance headers are only available for readable exports."""

    def __init__(self, output_format: str) -> None:
        self.output_format = output_format
        super().__init__(f"--provenance requires txt or md, got {output_format!r}")


class OutputExistsError(ExportError):
    """The destination already exists and replacement was not authorized."""

    def __init__(self, output: Path, *, replaceable: bool = True) -> None:
        self.output = Path(output)
        self.replaceable = replaceable
        super().__init__(f"output already exists: {self.output}")


class UnsafeOutputError(ExportError):
    """The destination resolves to an input transcript or canonical source."""

    def __init__(self, output: Path, protected: Path) -> None:
        self.output = Path(output)
        self.protected = Path(protected)
        super().__init__(f"output {self.output} resolves to protected input {self.protected}")


class OutputWriteError(ExportError):
    """The destination could not be published as a regular UTF-8 file."""

    def __init__(self, output: Path, reason: str) -> None:
        self.output = Path(output)
        self.reason = reason
        super().__init__(f"could not write output {self.output}: {reason}")
