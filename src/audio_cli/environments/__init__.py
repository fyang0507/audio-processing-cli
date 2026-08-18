"""Provisioned environments and the packages that target them.

`manifest.json` beside this module is the data; this is the only thing that reads it. It
answers three questions and nothing else:

- which environments exist, what interpreter each wants, and which lock materializes it;
- which packages exist, where each comes from, and what it costs to download;
- which packages a request needs, which is what `audio packages pull --stack S --want ...`
  and `run`'s exit-3 check both dispatch on.

The capability half of the stack table — availability, satisfaction, evidence, notes — is not
here. It belongs to the planner, and duplicating a column across the two is how two copies of
one fact start to disagree. `packages_for` is the seam: it takes a resolved role set rather
than raw capability names, so the planner owns the derivation and this module owns the
mapping to packages.

Nothing in this module touches the network, the filesystem outside the package directory, or
a provisioned environment. It describes what should exist; provisioning acts on it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"

#: The role enumeration, fixed and small (VOCABULARY.md, "role"). Deterministic glue is not a
#: role, so reconciling diarizer turns onto ASR text does not appear here.
ROLES = ("decode", "vad", "diarizer", "asr", "aligner", "punctuator", "lid")


class ManifestError(RuntimeError):
    """The manifest disagrees with itself, or with the files it names."""


@dataclass(frozen=True)
class Environment:
    name: str
    provisioned: bool
    python: str | None
    lock: Path | None
    requirements: Path | None
    reason: str
    provisional: bool
    requires_tool: tuple[str, ...]
    guards: tuple[dict, ...]

    @property
    def has_interpreter(self) -> bool:
        return self.python is not None


@dataclass(frozen=True)
class Package:
    id: str
    environment: str
    kind: str
    roles: tuple[str, ...]
    stacks: tuple[str, ...]
    bytes: int | None
    license_declared: str | None
    license_reviewed: bool
    source: dict
    checkout: dict | None
    requires_tool: tuple[str, ...]
    auto_fetch: bool
    note: str | None

    @property
    def sized(self) -> bool:
        return self.bytes is not None


@cache
def _raw() -> dict:
    document = json.loads(MANIFEST.read_text())
    version = document.get("schema_version")
    if version != 1:
        raise ManifestError(f"unsupported manifest schema_version {version!r}")
    return document


@cache
def environments() -> dict[str, Environment]:
    out: dict[str, Environment] = {}
    for name, body in _raw()["environments"].items():
        lock = body.get("lock")
        requirements = body.get("requirements")
        out[name] = Environment(
            name=name,
            provisioned=bool(body.get("provisioned")),
            python=body.get("python"),
            lock=HERE / lock if lock else None,
            requirements=HERE / requirements if requirements else None,
            reason=body.get("reason", ""),
            provisional=bool(body.get("provisional")),
            requires_tool=tuple(body.get("requires_tool", ())),
            guards=tuple(body.get("guards", ())),
        )
    return out


@cache
def packages() -> dict[str, Package]:
    out: dict[str, Package] = {}
    for identifier, body in _raw()["packages"].items():
        out[identifier] = Package(
            id=identifier,
            environment=body["environment"],
            kind=body["kind"],
            roles=tuple(body.get("roles", ())),
            stacks=tuple(body.get("stacks", ())),
            bytes=body.get("bytes"),
            license_declared=body.get("license_declared"),
            license_reviewed=bool(body.get("license_reviewed")),
            source=body["source"],
            checkout=body.get("checkout"),
            requires_tool=tuple(body.get("requires_tool", ())),
            auto_fetch=bool(body.get("auto_fetch")),
            note=body.get("note"),
        )
    return out


def packages_for(stack: str, roles: dict[str, str]) -> list[Package]:
    """Packages a resolved plan needs.

    `roles` maps a role to the backend filling it, which is what the planner produces. The
    lookup is by backend id rather than by capability, so this module never has to know why a
    role is in the plan — only which package supplies it.

    `decode` is absent from the result on purpose: ffmpeg is an external binary, reported by
    `audio doctor` rather than provisioned.
    """
    catalog = packages()
    unknown = sorted(set(roles.values()) - set(catalog) - {"ffmpeg"})
    if unknown:
        raise ManifestError(f"no package supplies backend(s) {unknown} for stack {stack!r}")

    wanted: dict[str, Package] = {}
    for role, backend in roles.items():
        if backend == "ffmpeg":
            continue
        package = catalog[backend]
        if role not in package.roles:
            raise ManifestError(
                f"{backend!r} does not fill the {role!r} role; manifest lists {package.roles}"
            )
        wanted[package.id] = package
        # A Core ML model package rides along with the toolchain that loads it.
        if package.id == "fluidaudio":
            wanted["speaker-diarization-coreml"] = catalog["speaker-diarization-coreml"]
    return sorted(wanted.values(), key=lambda item: item.id)


def download_bytes(selection: list[Package]) -> tuple[int, list[str]]:
    """Total known bytes, and the ids whose size is unknown.

    Two return values rather than one, because a plan that summed the known sizes and stayed
    silent about the rest would understate a download without saying so. `unsized_packages`
    exists for exactly this.
    """
    known = sum(package.bytes or 0 for package in selection)
    unsized = [package.id for package in selection if not package.sized]
    return known, unsized


def environments_spanned(selection: list[Package]) -> list[str]:
    """Provisioned environments a selection touches, in no meaningful order.

    `core` is excluded: it is the tool's own environment, so spanning it is not a cost.
    """
    catalog = environments()
    names = {package.environment for package in selection}
    return sorted(name for name in names if catalog[name].provisioned)


def lock_digest(environment: str) -> str | None:
    """sha256 of an environment's lock, or None where an environment has no interpreter."""
    lock = environments()[environment].lock
    if lock is None:
        return None
    return hashlib.sha256(lock.read_bytes()).hexdigest()


def validate() -> list[str]:
    """Problems with the manifest, as a list rather than an exception.

    Called by the test suite and by `audio doctor`. It is deliberately exhaustive rather than
    fail-fast: a reader fixing the manifest wants every problem at once.
    """
    problems: list[str] = []
    known_environments = environments()
    for package in packages().values():
        if package.environment not in known_environments:
            problems.append(f"{package.id}: unknown environment {package.environment!r}")
        for role in package.roles:
            if role not in ROLES:
                problems.append(f"{package.id}: {role!r} is not a role")
        if package.kind not in {"weights", "toolchain"}:
            problems.append(f"{package.id}: unexpected kind {package.kind!r}")
        if package.source.get("type") == "huggingface_multi":
            declared = sum(repo.get("bytes") or 0 for repo in package.source["repos"])
            if package.bytes != declared:
                problems.append(
                    f"{package.id}: bytes {package.bytes} does not equal the sum of its "
                    f"repos' bytes {declared}"
                )

    for environment in known_environments.values():
        if environment.provisioned and environment.has_interpreter:
            if environment.lock is None:
                problems.append(f"{environment.name}: provisioned with an interpreter but no lock")
            elif not environment.lock.is_file():
                problems.append(f"{environment.name}: lock {environment.lock.name} is missing")
        if not environment.provisioned and environment.lock is not None:
            problems.append(f"{environment.name}: not provisioned but names a lock")
        members = [p for p in packages().values() if p.environment == environment.name]
        if environment.provisioned and not members:
            problems.append(f"{environment.name}: provisioned but no package targets it")
    return problems
