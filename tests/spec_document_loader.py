"""Read the ordered fragments behind the transcription specification indexes."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_SPEC_PARTS = {
    (REPO / "TRANSCRIBE_HAPPY_PATH.md").resolve(): (
        REPO / "TRANSCRIBE_HAPPY_PATH_00_OVERVIEW_AND_MACHINE.md",
        REPO / "TRANSCRIBE_HAPPY_PATH_10_INTERVIEW_PLAN_AND_PROVISION.md",
        REPO / "TRANSCRIBE_HAPPY_PATH_20_INTERVIEW_RUN_AND_EXPORT.md",
        REPO / "TRANSCRIBE_HAPPY_PATH_30_VIDEO_EDITING.md",
        REPO / "TRANSCRIBE_HAPPY_PATH_40_DIALECT_RECORDING.md",
        REPO / "TRANSCRIBE_HAPPY_PATH_50_CORRECTIVE_PATHS.md",
        REPO / "TRANSCRIBE_HAPPY_PATH_60_TEARDOWN.md",
    ),
    (REPO / "TRANSCRIBE_CONTRACT.md").resolve(): (
        REPO / "TRANSCRIBE_CONTRACT_00_OVERVIEW.md",
        REPO / "TRANSCRIBE_CONTRACT_10_MACHINE_AND_QWEN.md",
        REPO / "TRANSCRIBE_CONTRACT_20_NATIVE_STACKS.md",
        REPO / "TRANSCRIBE_CONTRACT_30_EXPORT.md",
        REPO / "TRANSCRIBE_CONTRACT_40_REFUSALS.md",
        REPO / "TRANSCRIBE_CONTRACT_50_TEARDOWN_AND_COVERAGE.md",
    ),
    (REPO / "VOCABULARY.md").resolve(): (
        REPO / "VOCABULARY_00_TERMS.md",
        REPO / "VOCABULARY_10_FLOORS_CAPABILITIES_AND_RESOLUTION.md",
        REPO / "VOCABULARY_20_PACKAGES_AND_ENVIRONMENTS.md",
        REPO / "VOCABULARY_30_RETIRED_WORDS_AND_VERSIONING.md",
    ),
}


def spec_document_parts(path: Path) -> tuple[Path, ...]:
    """Return a specification's fragments in their normative order."""
    try:
        return _SPEC_PARTS[path.resolve()]
    except KeyError as error:
        raise ValueError(f"{path} is not a fragmented specification") from error


def read_spec_document(path: Path) -> str:
    """Return the complete specification, excluding its navigation-only root index."""
    return "".join(part.read_text() for part in spec_document_parts(path))
