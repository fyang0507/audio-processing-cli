"""Provisioning state, constants, and structured failures."""

from __future__ import annotations

from dataclasses import dataclass

REGISTRY_SCHEMA_VERSION = 1

HUB_CACHE_NOTE = (
    "weights live in the shared Hugging Face cache, not under this root. Only revisions this "
    "root recorded as downloaded and the current manifest still pins for that package are "
    "eligible for deletion; pre-existing and out-of-manifest revisions are retained because "
    "they may belong to another tool, another provisioning root, or an earlier experiment"
)
UNTOUCHED = ["user media", "transcript and subtitle outputs"]


@dataclass(frozen=True)
class CheckoutState:
    """Live Git state for a source checkout used by an executable backend."""

    head: str
    modified: tuple[str, ...]
    untracked: tuple[str, ...]


class ProvisioningError(RuntimeError):
    """A provisioning failure carrying the payload documented for its exit code."""

    def __init__(self, code: str, message: str, *, exit_code: int = 3, **payload) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code
        self.payload = payload

    def as_dict(self) -> dict:
        body = {"code": self.code, "detail": self.message}
        body.update(self.payload)
        return body


__all__ = [
    "CheckoutState",
    "HUB_CACHE_NOTE",
    "ProvisioningError",
    "REGISTRY_SCHEMA_VERSION",
    "UNTOUCHED",
]
