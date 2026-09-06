"""Read the ordered fragments behind the transcription specification indexes."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"

_SPEC_PARTS = {
    (REPO / "docs" / "TRANSCRIBE_HAPPY_PATH.md").resolve(): (
        DOCS / "transcribe-happy-path" / "00-overview-and-machine.md",
        DOCS / "transcribe-happy-path" / "10-interview-plan-and-provision.md",
        DOCS / "transcribe-happy-path" / "20-interview-run-and-export.md",
        DOCS / "transcribe-happy-path" / "30-video-editing.md",
        DOCS / "transcribe-happy-path" / "40-dialect-recording.md",
        DOCS / "transcribe-happy-path" / "50-corrective-paths.md",
        DOCS / "transcribe-happy-path" / "60-teardown.md",
    ),
    (REPO / "docs" / "TRANSCRIBE_CONTRACT.md").resolve(): (
        DOCS / "transcribe-contract" / "00-overview.md",
        DOCS / "transcribe-contract" / "10-machine-and-qwen.md",
        DOCS / "transcribe-contract" / "20-native-stacks.md",
        DOCS / "transcribe-contract" / "30-export.md",
        DOCS / "transcribe-contract" / "40-refusals.md",
        DOCS / "transcribe-contract" / "50-teardown-and-coverage.md",
    ),
    (REPO / "docs" / "VOCABULARY.md").resolve(): (
        DOCS / "vocabulary" / "00-terms.md",
        DOCS / "vocabulary" / "10-floors-capabilities-and-resolution.md",
        DOCS / "vocabulary" / "20-packages-and-environments.md",
        DOCS / "vocabulary" / "30-retired-words-and-versioning.md",
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
