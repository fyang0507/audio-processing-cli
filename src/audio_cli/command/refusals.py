"""Shared fixed-shape refusal primitives for command handlers.

Newer commands print these payloads bare on stderr. Older shipped commands retain their
historical ``{"error": ...}`` envelope, so this module is intentionally not a universal error
base for the whole CLI.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class Refusal(RuntimeError):
    """A structured command refusal that is printed bare on stderr."""

    def __init__(self, payload: Mapping[str, Any], *, exit_code: int) -> None:
        self.payload = dict(payload)
        self.exit_code = exit_code
        super().__init__(str(self.payload["code"]))


def build_refusal(code: str, exit_code: int, fix: str, **fields: Any) -> Refusal:
    return Refusal({"code": code, **fields, "fix": fix}, exit_code=exit_code)


def output_exists(output: str | Path, existing: str | Path, fix: str) -> Refusal:
    return build_refusal(
        "output_exists",
        2,
        fix,
        field="--output",
        provided=str(output),
        existing=str(existing),
    )


def output_is_canonical_input(
    output: str | Path,
    resolved_target: str | Path,
) -> Refusal:
    return build_refusal(
        "output_is_canonical_input",
        2,
        (
            "choose an --output that does not resolve to an input transcript, its derived "
            "partial path, or canonical source media; --force cannot override this"
        ),
        field="--output",
        provided=str(output),
        resolved_target=str(resolved_target),
    )


def output_path_invalid(
    output: str | Path,
    target: str | Path,
    reason: str,
) -> Refusal:
    return build_refusal(
        "output_path_invalid",
        2,
        (
            "choose an --output whose destination and parent directory can be "
            "resolved and written safely"
        ),
        field="--output",
        provided=str(output),
        target=str(target),
        reason=reason,
    )
