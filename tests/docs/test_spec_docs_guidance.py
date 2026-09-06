"""Vocabulary, examples, and shipped-guidance invariants for transcription specs."""

from __future__ import annotations

import re

from tests.docs.spec_document_loader import read_spec_document
from tests.docs.spec_document_test_support import (
    AGENT_GUIDANCE,
    CAPABILITY_NAMES,
    REPO,
    SHIPPED_SKILL_GUIDANCE,
    SPEC_DOCS,
    VOCABULARY,
)


def test_want_arguments_in_examples_use_real_capability_names() -> None:
    """A stale name in a shell example is as misleading as one in a payload."""
    intentionally_invalid = {"word_timing"}  # the capability_unknown demonstration
    for path in SPEC_DOCS:
        for line in read_spec_document(path).splitlines():
            # A prose description of --want is not an executable argument example.
            if not (
                line.lstrip().startswith(("audio ", "--want "))
                or re.search(r'"(?:fix|next)":\s*"audio ', line)
            ):
                continue
            match = re.search(r"--want ([a-z_][a-z_,]*)", line)
            if not match:
                continue  # `--want <capabilities>` placeholders carry no names to check
            for name in filter(None, match.group(1).split(",")):
                assert name in CAPABILITY_NAMES or name in intentionally_invalid, (
                    f"{path.name}: --want {name!r} is not a capability name"
                )


def test_vocabulary_publishes_the_namespace_the_examples_use() -> None:
    """The naming contract and the worked examples must not drift apart."""
    text = read_spec_document(VOCABULARY)
    for name in CAPABILITY_NAMES:
        assert f"`{name}`" in text, f"VOCABULARY.md does not define {name!r}"


def test_vibevoice_cap_projection_is_scoped_and_labelled_in_both_specs() -> None:
    for path in SPEC_DOCS:
        text = " ".join(read_spec_document(path).split())
        assert "11,345 generated tokens over 1,800 seconds" in text
        assert "declared 16,384-token cap" in text
        assert "about 43 minutes of comparable audio" in text
        assert "rate extrapolation, not an observed truncation" in text


def test_repository_and_shipped_skill_describe_the_current_run_surface() -> None:
    """The packaged user instructions must advance when an executable phase ships."""
    stale_claims = (
        "only the two qwen",
        "only the qwen",
        "run adapters have not shipped",
        "remain work in progress",
        "export are issues #22",
        "export is not shipped",
        "export are not shipped",
    )
    for path in (AGENT_GUIDANCE, *SHIPPED_SKILL_GUIDANCE):
        text = path.read_text().lower()
        for claim in stale_claims:
            assert claim not in text, f"{path.relative_to(REPO)} retains stale claim {claim!r}"

    repository_text = AGENT_GUIDANCE.read_text()
    assert "transcribe through four explicit stacks" in repository_text
    assert "`transcribe run`, and `export` ship" in repository_text

    # The skill routes to live help/capabilities instead of duplicating stack catalogs.
    # Fresh-context acceptance evaluates its guidance; literal backend lists are not required.
