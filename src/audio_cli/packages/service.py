"""Explicit package-provisioning service with injectable external boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..environments import Package
from . import lifecycle, pull, verify
from .fetcher import Fetcher
from .toolchain import Toolchain


@dataclass
class Provisioner:
    """Coordinate package workflows without hiding behavior in an inheritance graph."""

    toolchain: Toolchain = field(default_factory=Toolchain)
    fetcher: Fetcher = field(default_factory=Fetcher)

    def ensure_environment(self, name: str, document: dict) -> bool:
        return pull.ensure_environment(self.toolchain, name, document)

    def pull(
        self,
        selection: list[Package],
        *,
        repair: bool = False,
        stack: str | None = None,
    ) -> dict:
        return pull.pull(
            self.toolchain,
            self.fetcher,
            selection,
            repair=repair,
            stack=stack,
        )

    def verify(self, *, repair: bool = False) -> dict:
        return verify.verify(self.toolchain, repair=repair)

    def remove(self, package_ids: list[str]) -> dict:
        return lifecycle.remove(self.fetcher, package_ids)

    def purge(self, *, dry_run: bool) -> dict:
        return lifecycle.purge(self.fetcher, dry_run=dry_run)


__all__ = ["Provisioner"]
