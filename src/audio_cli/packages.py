"""`audio packages` — provisioning, verification, and teardown.

The rules this implements are in VOCABULARY.md and the mechanism is in ENVIRONMENTS.md. Three
of them shape every function here:

- **Fails closed.** `pull` is the only thing that downloads weights, builds the Swift product,
  or applies a patch. A transcription request never triggers any of them; it reports exit 3 with
  the `pull` line that would fix it.
- **A crashed `pull` cannot read as provisioned.** An entry is written with `state` set to
  `pulling` or `creating` *before* any bytes move and flipped to `ready` only after its check
  passes. Anything not `ready` counts as absent to a run and as reclaimable to `purge`.
  Recording intent first is what makes a half-finished download nameable at all.
- **Reference counts are derived, never stored.** An environment survives exactly while some
  non-absent package targets it. A stored count would eventually disagree with the table it
  summarizes.

The two external surfaces — running `uv`/`swift`/`git`, and downloading — are injected, so the
registry, the state machine, reference counting, and every payload shape are testable without a
network or a toolchain.
"""

from __future__ import annotations

import hashlib
import fnmatch
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import paths
from .media import (
    assert_directory_binding,
    bound_directory,
    cleanup_temporary_file,
    file_identity_from_descriptor,
    publish_temporary_file,
    sha256_regular_file_at,
)
from .environments import (
    HERE as ENVIRONMENTS_DIR,
    Environment,
    ManifestError,
    Package,
    environments,
    packages,
)

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
    """A provisioning failure that carries the payload its exit code is documented with."""

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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_bytes(path: Path) -> int:
    """Total size under a directory, following symlinks into the Hub's blob store."""
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:  # a broken symlink contributes nothing rather than failing the pull
            continue
    return total


def _hub_snapshot_index() -> dict[tuple[str, str], Path]:
    """Map canonical Hub cache identity to the snapshot path reported by its cache index."""
    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir()
    return {
        (repository.repo_id, revision.commit_hash): Path(revision.snapshot_path)
        for repository in cache.repos
        for revision in repository.revisions
    }


def _inspect_hub_snapshot(
    repository: str,
    revision: str,
    snapshot: Path | None,
    patterns: tuple[str, ...],
    snapshot_index: Mapping[tuple[str, str], Path],
) -> tuple[list[str], int | None]:
    """Bind one returned Hub path before reading any bytes below it."""
    if snapshot is None:
        return [f"{repository} snapshot is not a non-symlink directory: {snapshot}"], None
    expected = snapshot_index.get((repository, revision))
    if expected is None:
        return [
            f"{repository} revision {revision} is absent from the Hugging Face cache index"
        ], None
    # Refuse a caller-returned path that is not lexically the indexed path before
    # resolving or walking it.  A bad downloader return must not make pull inspect an
    # unrelated tree merely to discover that it was unrelated.
    if Path(os.path.abspath(snapshot)) != Path(os.path.abspath(expected)):
        return [
            f"{repository} snapshot path {snapshot} does not equal cache-indexed "
            f"revision path {expected}"
        ], None
    if (
        expected.is_symlink()
        or expected.parent.is_symlink()
        or expected.parent.parent.is_symlink()
    ):
        return [f"{repository} snapshot is not the exact non-symlink cache-index path"], None
    if not expected.is_dir():
        return [
            f"{repository} snapshot is not a non-symlink directory: {expected}"
        ], None
    try:
        expected_resolved = expected.resolve(strict=True)
        repository_cache_root = expected.parent.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return [f"{repository} indexed snapshot cannot be resolved: {exc}"], None

    unsafe_entries: list[str] = []
    snapshot_files: list[str] = []
    try:
        for entry in expected_resolved.rglob("*"):
            if entry.is_symlink():
                target = entry.resolve(strict=True)
                if not target.is_file() or not target.is_relative_to(
                    repository_cache_root
                ):
                    unsafe_entries.append(str(entry.relative_to(expected_resolved)))
            elif entry.is_file():
                target = entry.resolve(strict=True)
                if not target.is_relative_to(repository_cache_root):
                    unsafe_entries.append(str(entry.relative_to(expected_resolved)))
            if entry.is_file():
                snapshot_files.append(entry.relative_to(expected_resolved).as_posix())
    except (OSError, RuntimeError) as exc:
        return [f"{repository} snapshot tree cannot be resolved safely: {exc}"], None
    if unsafe_entries:
        return [
            f"{repository} snapshot entries resolve outside its repository cache: "
            f"{sorted(unsafe_entries)!r}"
        ], None

    issues = [
        f"{repository} is missing allow_pattern {pattern}"
        for pattern in patterns
        if not any(fnmatch.fnmatch(name, pattern) for name in snapshot_files)
    ]
    if issues:
        return issues, None
    try:
        return [], _tree_bytes(expected_resolved)
    except OSError as exc:
        return [f"{repository} snapshot bytes cannot be measured safely: {exc}"], None


def hub_materialization_issues(
    package: Package, materialized: dict,
) -> list[str]:
    """Return cheap live integrity failures for revision-pinned Hub snapshots."""
    kind = package.source["type"]
    if kind not in {"huggingface", "huggingface_multi"}:
        return []

    snapshots: list[tuple[str, str, Path | None, tuple[str, ...]]] = []
    if kind == "huggingface":
        value = materialized.get("path")
        snapshots.append((
            package.source["repo"],
            package.source["revision"],
            Path(str(value)) if value else None,
            tuple(package.source.get("allow_patterns", ())),
        ))
    else:
        values = materialized.get("paths")
        values = values if isinstance(values, dict) else {}
        for repository in package.source["repos"]:
            value = values.get(repository["repo"])
            snapshots.append((
                repository["repo"],
                repository["revision"],
                Path(str(value)) if value else None,
                tuple(repository.get("allow_patterns", ())),
            ))

    try:
        snapshot_index = _hub_snapshot_index()
    except Exception as exc:  # noqa: BLE001 - no cache identity means no trusted snapshot
        return [f"Hugging Face cache identity cannot be inspected: {exc}"]
    issues: list[str] = []
    actual_bytes = 0
    all_usable = True
    for repository, revision, snapshot, patterns in snapshots:
        snapshot_issues, byte_count = _inspect_hub_snapshot(
            repository, revision, snapshot, patterns, snapshot_index
        )
        issues.extend(snapshot_issues)
        if byte_count is None:
            all_usable = False
        else:
            actual_bytes += byte_count
    recorded_bytes = materialized.get("bytes")
    if isinstance(recorded_bytes, bool) or not isinstance(recorded_bytes, int):
        issues.append("materialization receipt has no integer bytes measurement")
    elif all_usable and actual_bytes != recorded_bytes:
        issues.append(
            f"snapshot bytes changed: recorded {recorded_bytes}, current {actual_bytes}"
        )
    return issues


# --------------------------------------------------------------------------------------
# Injected external surfaces
# --------------------------------------------------------------------------------------


@dataclass
class Toolchain:
    """The external commands provisioning shells out to."""

    def which(self, tool: str) -> str | None:
        return shutil.which(tool)

    def run(self, args: list[str], *, cwd: Path | None = None,
            timeout: int = 3600) -> subprocess.CompletedProcess:
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)

    def file_digest(self, path: Path) -> str:
        """Hash a checkout file; injectable so tests can model pinned upstream bytes."""
        return sha256_file(path)

    def create_environment(self, environment: Environment, target: Path) -> None:
        """`uv venv` then `uv pip sync` — no resolution at provisioning time, ever."""
        if self.which("uv") is None:
            raise ProvisioningError(
                "toolchain_missing", "uv is not on PATH and is required to create environments",
                missing_tool="uv", fix="Install uv: https://docs.astral.sh/uv/",
            )
        created = self.run(["uv", "venv", "--python", environment.python, str(target)])
        if created.returncode != 0:
            raise ProvisioningError(
                "environment_creation_failed",
                f"could not create {environment.name}: {created.stderr.strip()}",
                environment=environment.name,
            )
        synced = self.run(["uv", "pip", "sync", "--python", str(target / "bin" / "python"),
                           str(environment.lock)])
        if synced.returncode != 0:
            raise ProvisioningError(
                "environment_sync_failed",
                f"could not sync {environment.name} from its lock: {synced.stderr.strip()}",
                environment=environment.name, lock=environment.lock.name,
            )

    def frozen_packages(self, environment_python: Path) -> dict[str, str]:
        result = self.run(["uv", "pip", "freeze", "--python", str(environment_python)])
        if result.returncode != 0:
            return {}
        frozen: dict[str, str] = {}
        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-e "):
                source = line[3:].strip()
                try:
                    fragment = urllib.parse.urlsplit(source).fragment
                except ValueError:
                    fragment = ""
                egg = urllib.parse.parse_qs(fragment).get("egg", [])
                name = _distribution_name(egg[0]) if egg else f"<editable:{source}>"
                frozen[name] = f"-e {source}"
            elif " @ " in line:
                name, _, source = line.partition(" @ ")
                frozen[_distribution_name(name)] = f"@ {source.strip()}"
            elif "==" in line:
                name, _, version = line.partition("==")
                frozen[_distribution_name(name)] = version.strip()
            else:
                # Unknown freeze syntax is installed state too. A stable synthetic name makes
                # equality fail closed instead of silently discarding the distribution.
                frozen[f"<unparsed:{line}>"] = line
        return frozen

    def clone(self, repo: str, commit: str, target: Path) -> None:
        if self.which("git") is None:
            raise ProvisioningError(
                "toolchain_missing", "git is not on PATH and is required for source checkouts",
                missing_tool="git",
            )
        if not target.exists():
            cloned = self.run(["git", "clone", "--quiet", repo, str(target)])
            if cloned.returncode != 0:
                raise ProvisioningError("checkout_failed",
                                        f"could not clone {repo}: {cloned.stderr.strip()}")
        checked = self.run(["git", "checkout", "--quiet", commit], cwd=target)
        if checked.returncode != 0:
            raise ProvisioningError("checkout_failed",
                                    f"could not check out {commit}: {checked.stderr.strip()}")

    def apply_patch(self, checkout: Path, patch: Path) -> None:
        applied = self.run(["git", "apply", str(patch)], cwd=checkout)
        if applied.returncode != 0:
            # Already applied is not a failure: pull must be repeatable.
            check = self.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=checkout)
            if check.returncode != 0:
                raise ProvisioningError(
                    "patch_failed",
                    f"could not apply {patch.name}: {applied.stderr.strip()}",
                    patch=patch.name,
                )

    def install_checkout(self, environment_python: Path, checkout: Path) -> None:
        result = self.run(["uv", "pip", "install", "--python", str(environment_python),
                           "--no-deps", str(checkout)])
        if result.returncode != 0:
            raise ProvisioningError(
                "checkout_install_failed",
                f"could not install {checkout.name} into the environment: "
                f"{result.stderr.strip()}",
            )

    def clean_ignored_checkout(self, checkout: Path) -> None:
        """Remove install-time ignored artifacts before the checkout becomes executable."""
        result = self.run(["git", "clean", "-fdX"], cwd=checkout)
        if result.returncode != 0:
            raise ProvisioningError(
                "checkout_cleanup_failed",
                f"could not remove ignored build artifacts from {checkout.name}: "
                f"{result.stderr.strip()}",
            )

    def inspect_checkout(self, checkout: Path) -> CheckoutState:
        """Read HEAD plus every provenance-relevant tracked and untracked path.

        Ignored bytecode and extension modules can still be imported from a checkout, so the
        second ``ls-files`` invocation is intentional.  A default ``git status`` would hide
        exactly the files this check exists to catch.
        """

        def git(*arguments: str) -> str:
            try:
                result = self.run(["git", *arguments], cwd=checkout)
            except OSError as exc:
                raise ValueError(f"could not inspect source checkout: {exc}") from exc
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise ValueError(
                    f"git {' '.join(arguments)} failed for {checkout}: {detail}"
                )
            return result.stdout

        head = git("rev-parse", "--verify", "HEAD^{commit}").strip()
        modified = tuple(sorted(filter(None, git(
            "diff", "--name-only", "--no-ext-diff", "--no-textconv", "--no-renames",
            "HEAD", "--",
        ).splitlines())))
        ordinary_untracked = filter(None, git(
            "ls-files", "--others", "--exclude-standard",
        ).splitlines())
        ignored_untracked = filter(None, git(
            "ls-files", "--others", "--ignored", "--exclude-standard",
        ).splitlines())
        return CheckoutState(
            head=head,
            modified=modified,
            untracked=tuple(sorted({*ordinary_untracked, *ignored_untracked})),
        )

    def swift_build(self, checkout: Path) -> None:
        if self.which("swift") is None:
            raise ProvisioningError(
                "toolchain_missing",
                "swift is not on PATH; it is required only for the swift environment's packages",
                missing_tool="swift", requires_tool=["swift"],
            )
        built = self.run(["swift", "build", "-c", "release"], cwd=checkout)
        if built.returncode != 0:
            raise ProvisioningError("swift_build_failed",
                                    f"swift build failed: {built.stderr.strip()[-600:]}")

    def swift_product_runs(self, checkout: Path, product: str) -> bool:
        """Whether the built executable actually launches.

        The product name is pinned in the manifest rather than written here. It was `fluidaudio`
        in this call, and `Package.swift` declares `.executable(name: "fluidaudiocli")` -- so
        `swift run` answered `no executable product named 'fluidaudio'` on a perfectly good build
        and this check could never return True. A name that lives beside the commit it belongs to
        is reviewable when the commit moves; a literal buried in a runner is not.
        """
        result = self.run(["swift", "run", "-c", "release", product, "--help"],
                          cwd=checkout, timeout=600)
        return result.returncode == 0

    def built_product_runs(self, executable: Path) -> bool:
        """Probe an already-built product without requiring its provisioning toolchain."""
        result = self.run([str(executable), "--help"], cwd=executable.parent, timeout=600)
        return result.returncode == 0


@dataclass
class Fetcher:
    """Downloads. Hub weights stay in the Hugging Face cache; the registry records revisions."""

    def cached_revisions(self) -> set[str]:
        """Revisions the Hub cache already holds, whoever put them there."""
        try:
            from huggingface_hub import scan_cache_dir
        except ImportError:
            return set()
        try:
            cache = scan_cache_dir()
        except Exception:  # noqa: BLE001 - an unreadable cache holds nothing we can claim
            return set()
        return {revision.commit_hash for repo in cache.repos for revision in repo.revisions}

    def hf_snapshot(
        self,
        repo: str,
        revision: str,
        *,
        force: bool = False,
        allow_patterns: tuple[str, ...] | None = None,
    ) -> Path:
        """Materialize a revision into the shared Hub cache, resuming a partial download.

        `force` re-downloads what the cache already holds, and it is what makes `pull --repair`
        mean anything for a Hub package: without it `snapshot_download` sees the revision present
        and returns the snapshot unchanged, so a repair of a corrupt snapshot would report success
        having moved no bytes.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # noqa: BLE001 - reported, not swallowed
            raise ProvisioningError(
                "toolchain_missing",
                "huggingface_hub is required to download weights",
                missing_tool="huggingface_hub",
                fix="uv pip install huggingface_hub",
            ) from exc
        return Path(snapshot_download(
            repo,
            revision=revision,
            force_download=force,
            allow_patterns=list(allow_patterns) if allow_patterns is not None else None,
        ))

    def delete_hub_revisions(self, revisions: list[str]) -> tuple[list[str], int]:
        """Delete exactly these revisions from the Hub cache. Returns what went, and its size.

        Revision-scoped rather than repo-scoped: the cache may be shared with other tools, and
        another revision of the same repository is not ours to remove. Revisions the cache no
        longer holds are reported as not deleted rather than as deleted, because a teardown
        report that overstates what it freed is worse than one that reclaims less.
        """
        if not revisions:
            return [], 0
        try:
            from huggingface_hub import scan_cache_dir
        except ImportError:
            return [], 0
        try:
            cache = scan_cache_dir()
        except Exception:  # noqa: BLE001 - a missing or unreadable cache frees nothing
            return [], 0
        present = {
            revision.commit_hash
            for repo in cache.repos
            for revision in repo.revisions
        }
        deletable = sorted(set(revisions) & present)
        if not deletable:
            return [], 0
        strategy = cache.delete_revisions(*deletable)
        freed = int(strategy.expected_freed_size)
        strategy.execute()
        return deletable, freed

    def url_file(self, url: str, sha256: str, target: Path) -> Path:
        """Hash and publish through one no-follow descriptor bound to the managed parent."""
        try:
            with bound_directory(
                target.parent,
                root=paths.root(),
                create=True,
            ) as parent_descriptor:
                try:
                    existing = sha256_regular_file_at(
                        parent_descriptor, target.name
                    )
                except OSError:
                    # An unreadable cache entry cannot earn the fast path. Replacement is
                    # descriptor-relative, so retrying the download does not follow it.
                    existing = None
                if existing == sha256:
                    return target
                partial_name = (
                    f".audio-download-{os.getpid()}-{uuid.uuid4().hex}.part"
                )
                created = False
                temporary_identity = None
                try:
                    descriptor = os.open(
                        partial_name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL
                        | getattr(os, "O_NOFOLLOW", 0)
                        | getattr(os, "O_CLOEXEC", 0),
                        0o666,
                        dir_fd=parent_descriptor,
                    )
                    created = True
                    try:
                        temporary_identity = file_identity_from_descriptor(
                            descriptor, target.parent / partial_name
                        )
                    finally:
                        if temporary_identity is None:
                            os.close(descriptor)
                    if temporary_identity is None:
                        raise OSError(
                            f"download temporary is not a regular file: "
                            f"{target.parent / partial_name}"
                        )
                    request = urllib.request.Request(
                        url, headers={"User-Agent": "audio-processing-cli/0.1"})
                    with os.fdopen(descriptor, "wb") as output:
                        response = urllib.request.urlopen(request, timeout=60)
                        with response:
                            digest = hashlib.sha256()
                            while chunk := response.read(1024 * 1024):
                                output.write(chunk)
                                digest.update(chunk)
                            output.flush()
                            os.fsync(output.fileno())
                    actual = digest.hexdigest()
                    if actual != sha256:
                        raise ProvisioningError(
                            "package_integrity_failed",
                            f"{target.name} checksum mismatch: expected {sha256}, got {actual}",
                            expected=sha256, actual=actual,
                        )
                    assert_directory_binding(parent_descriptor, target.parent)
                    publish_temporary_file(
                        parent_descriptor,
                        partial_name,
                        target.name,
                        output_path=target,
                        force=True,
                        temporary_identity=temporary_identity,
                        replace_non_directory=True,
                    )
                    created = False
                finally:
                    if created and temporary_identity is not None:
                        cleanup_temporary_file(
                            parent_descriptor, partial_name, temporary_identity
                        )
        except (OSError, urllib.error.URLError) as exc:
            raise ProvisioningError("download_failed",
                                    f"could not download {url}: {exc}") from exc
        return target


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------


def blank_registry() -> dict:
    from . import __version__

    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "tool_version": __version__,
        "root": str(paths.root()),
        "environments": {},
        "packages": {},
    }


def _reject_duplicate_registry_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def load_registry() -> dict:
    target = paths.registry_path()
    try:
        with bound_directory(
            target.parent,
            root=paths.root(),
            create=False,
        ) as parent_descriptor:
            try:
                state = os.stat(
                    target.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return blank_registry()
            if stat.S_ISLNK(state.st_mode):
                raise ProvisioningError(
                    "registry_unreadable",
                    f"{target} is a symlink, not this root's registry",
                    fix=f"Move {target} aside and run audio packages pull again",
                )
            if not stat.S_ISREG(state.st_mode):
                raise ProvisioningError(
                    "registry_unreadable",
                    f"{target} exists but is not a regular file",
                    fix=f"Move {target} aside and run audio packages pull again",
                )
            descriptor = os.open(
                target.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_descriptor,
            )
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                opened = os.fstat(handle.fileno())
                if not stat.S_ISREG(opened.st_mode):
                    raise ProvisioningError(
                        "registry_unreadable",
                        f"{target} exists but is not a regular file",
                        fix=f"Move {target} aside and run audio packages pull again",
                    )
                raw = handle.read()
    except FileNotFoundError:
        # An absent root is the initial, unprovisioned state.  It is different
        # from a present redirected or unreadable root, which fails below.
        return blank_registry()
    except ProvisioningError:
        raise
    except (OSError, UnicodeError) as exc:
        raise ProvisioningError(
            "registry_unreadable", f"could not read {target}: {exc}",
            fix=f"Restore access to {target}, or move it aside and run audio packages pull again",
        ) from exc
    try:
        document = json.loads(
            raw, object_pairs_hook=_reject_duplicate_registry_keys
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ProvisioningError(
            "registry_unreadable", f"{target} is not valid JSON: {exc}",
            fix=f"Move {target} aside and run audio packages pull again",
        ) from exc
    if not isinstance(document, dict):
        raise ProvisioningError(
            "registry_unreadable", f"{target} must contain a JSON object",
            fix=f"Move {target} aside and run audio packages pull again",
        )
    if document.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ProvisioningError(
            "registry_unreadable",
            f"{target} has schema_version {document.get('schema_version')!r}, expected "
            f"{REGISTRY_SCHEMA_VERSION}",
        )
    expected_root = str(paths.root())
    if document.get("root") != expected_root:
        raise ProvisioningError(
            "registry_unreadable",
            f"{target} records provisioning root {document.get('root')!r}, expected "
            f"{expected_root!r}",
            fix=f"Restore {target} from this root, or move it aside and pull again",
        )
    for key in ("environments", "packages"):
        value = document.setdefault(key, {})
        if not isinstance(value, dict) or any(
            not isinstance(identifier, str) or not isinstance(entry, dict)
            for identifier, entry in value.items()
        ):
            raise ProvisioningError(
                "registry_unreadable",
                f"{target} field {key!r} must be an object of object entries",
                fix=f"Move {target} aside and run audio packages pull again",
            )
        if key == "packages":
            malformed = sorted(
                identifier
                for identifier, entry in value.items()
                if "materialized" in entry
                and not isinstance(entry["materialized"], dict)
            )
            if malformed:
                raise ProvisioningError(
                    "registry_unreadable",
                    f"{target} package materialized fields must be objects: {malformed}",
                    fix=f"Move {target} aside and run audio packages pull again",
                )
            for identifier, entry in value.items():
                retry_revisions = entry.get("hub_revisions_pre_existing")
                if retry_revisions is not None and (
                    not isinstance(retry_revisions, list)
                    or any(not isinstance(revision, str) for revision in retry_revisions)
                ):
                    raise ProvisioningError(
                        "registry_unreadable",
                        f"{target} package {identifier!r} hub_revisions_pre_existing "
                        "must be an array of strings",
                        fix=f"Move {target} aside and run audio packages pull again",
                    )
                materialized = entry.get("materialized", {})
                byte_count = materialized.get("bytes")
                if byte_count is not None and (
                    isinstance(byte_count, bool)
                    or not isinstance(byte_count, int)
                    or byte_count < 0
                ):
                    raise ProvisioningError(
                        "registry_unreadable",
                        f"{target} package {identifier!r} materialized bytes must be a "
                        "non-negative integer",
                        fix=f"Move {target} aside and run audio packages pull again",
                    )
                for location_key in ("path", "checkout"):
                    location = materialized.get(location_key)
                    if location is not None and not isinstance(location, str):
                        raise ProvisioningError(
                            "registry_unreadable",
                            f"{target} package {identifier!r} materialized "
                            f"{location_key} must be a string",
                            fix=f"Move {target} aside and run audio packages pull again",
                        )
                locations = materialized.get("paths")
                if locations is not None and (
                    not isinstance(locations, dict)
                    or any(
                        not isinstance(repo, str) or not isinstance(location, str)
                        for repo, location in locations.items()
                    )
                ):
                    raise ProvisioningError(
                        "registry_unreadable",
                        f"{target} package {identifier!r} materialized paths must map "
                        "strings to strings",
                        fix=f"Move {target} aside and run audio packages pull again",
                    )
                for revisions_key in (
                    "hub_revisions", "hub_revisions_pre_existing",
                ):
                    revisions = materialized.get(revisions_key)
                    if revisions is not None and (
                        not isinstance(revisions, list)
                        or any(not isinstance(revision, str) for revision in revisions)
                    ):
                        raise ProvisioningError(
                            "registry_unreadable",
                            f"{target} package {identifier!r} materialized "
                            f"{revisions_key} must be an array of strings",
                            fix=f"Move {target} aside and run audio packages pull again",
                        )
    return document


def save_registry(document: dict) -> None:
    """Atomically publish through one descriptor bound to the managed root."""
    target = paths.registry_path()
    try:
        with bound_directory(
            target.parent,
            root=paths.root(),
            create=True,
        ) as parent_descriptor:
            try:
                existing = os.stat(
                    target.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                existing = None
            if existing is not None and not stat.S_ISREG(existing.st_mode):
                raise ProvisioningError(
                    "registry_unreadable",
                    f"{target} exists but is not a regular file",
                    fix=f"Move {target} aside and run audio packages pull again",
                )

            partial_name = (
                f".audio-registry-{os.getpid()}-{uuid.uuid4().hex}.tmp"
            )
            created = False
            temporary_identity = None
            try:
                descriptor = os.open(
                    partial_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o666,
                    dir_fd=parent_descriptor,
                )
                created = True
                try:
                    temporary_identity = file_identity_from_descriptor(
                        descriptor, target.parent / partial_name
                    )
                finally:
                    if temporary_identity is None:
                        os.close(descriptor)
                if temporary_identity is None:
                    raise OSError(
                        f"registry temporary is not a regular file: "
                        f"{target.parent / partial_name}"
                    )
                with os.fdopen(
                    descriptor, "w", encoding="utf-8", newline="\n"
                ) as handle:
                    handle.write(
                        json.dumps(
                            document,
                            indent=2,
                            sort_keys=True,
                            ensure_ascii=False,
                        ) + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())

                assert_directory_binding(parent_descriptor, target.parent)
                publish_temporary_file(
                    parent_descriptor,
                    partial_name,
                    target.name,
                    output_path=target,
                    force=True,
                    temporary_identity=temporary_identity,
                )
                created = False
            finally:
                if created and temporary_identity is not None:
                    cleanup_temporary_file(
                        parent_descriptor, partial_name, temporary_identity
                    )
    except ProvisioningError:
        raise
    except (OSError, UnicodeError) as exc:
        raise ProvisioningError(
            "registry_unreadable",
            f"could not write {target} safely: {exc}",
            fix=f"Restore access to {target}, or move it aside and pull again",
        ) from exc


def is_ready(document: dict, package_id: str) -> bool:
    return document.get("packages", {}).get(package_id, {}).get("state") == "ready"


def missing_packages(selection: list[Package]) -> list[Package]:
    """What `run` reports at exit 3. Anything not `ready` is absent, including a crashed pull."""
    document = load_registry()
    return [package for package in selection if not is_ready(document, package.id)]


# --------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------


def select(package_ids: list[str] | None = None, *, stack: str | None = None) -> list[Package]:
    """Packages named directly, or every package a stack can use.

    `pull --stack S --want ...` is meant to take its set from a plan. Until the planner exists
    (#12), a stack selects every package that lists it, which over-provisions rather than
    under-provisions — the failure that would matter here is a missing package at run time.

    The two forms are alternatives rather than layers, so passing both is refused instead of
    resolved by precedence. This function used to return the named packages and drop `--stack` on
    the floor, which is the same defect as a silently ignored `--want`: the caller reads a receipt
    for a set it did not ask for and cannot tell which input was honoured.
    """
    catalog = packages()
    if package_ids and stack is not None:
        raise ProvisioningError(
            "stack_conflicts_with_named_packages",
            f"--stack {stack} was passed alongside named packages; a stack selects every package "
            "it can use and named ids select exactly those, so one of the two has to go",
            exit_code=2, stack=stack, packages=list(package_ids),
            fix=f"audio packages pull {' '.join(package_ids)}",
        )
    if package_ids:
        unknown = sorted(set(package_ids) - set(catalog))
        if unknown:
            raise ProvisioningError(
                "package_unknown", f"no such package: {', '.join(unknown)}", exit_code=2,
                allowed=sorted(catalog),
            )
        # A repeated positional id is still one package selection.  Besides duplicating the
        # receipt, preserving duplicates is destructive under ``--repair``: the same weights are
        # force-downloaded or the same checkout is rebuilt twice.  ``dict`` retains the caller's
        # first-occurrence order while making the selection a stable set.
        return [catalog[identifier] for identifier in dict.fromkeys(package_ids)]
    if stack is not None:
        chosen = [package for package in catalog.values() if stack in package.stacks]
        if not chosen:
            raise ProvisioningError(
                "stack_unknown", f"no packages are registered for stack {stack!r}", exit_code=2,
                allowed=sorted({name for p in catalog.values() for name in p.stacks}),
            )
        return sorted(chosen, key=lambda package: package.id)
    raise ProvisioningError("nothing_selected", "name packages, or pass --stack", exit_code=2)


def _path_location_fields(entry: dict) -> dict:
    """The one location shape a materialization actually has; absent keys stay absent."""
    materialized = entry.get("materialized", {})
    locations = materialized.get("paths")
    fields: dict = {}
    if isinstance(locations, dict):
        fields["locations"] = dict(sorted(locations.items()))
    else:
        location = materialized.get("path")
        if location is not None:
            fields["location"] = location
    if materialized.get("checkout") is not None:
        fields["checkout"] = materialized["checkout"]
    return fields


# --------------------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------------------


def path_report() -> dict:
    """Where everything is, so a session with no provisioning history can still find it."""
    document = load_registry()
    catalog = packages()
    return {
        "root": str(paths.root()),
        "registry": str(paths.registry_path()),
        # Most of the bytes are not under the root, and a reader who assumes they are concludes
        # that a 17 GiB pull silently did nothing. Say where weights actually land, and say what
        # the models directory is for rather than printing a path that is often absent.
        "weights": {
            "location": "the Hugging Face cache, shared with other tools",
            "note": "per-package `location` or `locations` below is authoritative; the "
                    "registry records the revisions this root materialized there",
        },
        "models": {
            "path": str(paths.models_dir()),
            "exists": paths.models_dir().exists(),
            "holds": "hash-pinned single-file artifacts only, currently silero-vad",
        },
        "environments": {
            name: {"path": str(paths.env_dir(name)),
                   "python": str(paths.env_python(name)) if environment.has_interpreter else None,
                   "state": document["environments"].get(name, {}).get("state", "absent")}
            for name, environment in environments().items() if environment.provisioned
        },
        "packages": {
            identifier: {
                "state": entry.get("state", "absent"),
                **(
                    {"location": str(
                        paths.models_dir() / str(catalog[identifier].source["filename"])
                    )}
                    if identifier in catalog
                    and catalog[identifier].source["type"] == "url"
                    else _path_location_fields(entry)
                ),
            }
            for identifier, entry in sorted(document["packages"].items())
        },
    }


def list_report() -> dict:
    document = load_registry()
    catalog = packages()
    listed = []
    total_known = 0
    unsized: list[str] = []
    for identifier, entry in sorted(document["packages"].items()):
        package = catalog.get(identifier)
        size = entry.get("materialized", {}).get("bytes")
        if size is None and package is not None:
            size = package.bytes
        if size is None:
            unsized.append(identifier)
        else:
            total_known += size
        listed.append({
            "package": identifier,
            "environment": package.environment if package else entry.get("environment"),
            "state": entry.get("state"),
            "bytes": size,
            "license_declared": (
                package.license_declared if package else entry.get("license_declared")
            ),
            "license_reviewed": (
                package.license_reviewed
                if package else entry.get("license_reviewed", False)
            ),
            "used_by_stacks": list(package.stacks) if package else [],
        })
    return {
        "root": str(paths.root()),
        "packages": listed,
        "environments": {
            name: document["environments"].get(name, {}).get("state", "absent")
            for name, environment in environments().items() if environment.provisioned
        },
        "total_known_bytes": total_known,
        "unsized_packages": unsized,
    }


@dataclass
class Provisioner:
    toolchain: Toolchain = field(default_factory=Toolchain)
    fetcher: Fetcher = field(default_factory=Fetcher)

    # -- environments ------------------------------------------------------------------

    def ensure_environment(self, name: str, document: dict) -> bool:
        """Create an environment if it is not already `ready`. Returns whether it was created."""
        environment = environments()[name]
        if not environment.provisioned:
            return False
        target = paths.env_dir(name)
        target_issue = managed_environment_creation_target_issue(name)
        if target_issue is not None:
            raise ProvisioningError(
                "environment_drifted",
                target_issue,
                environment=name,
                fix=(
                    f"Replace the redirected environment path {target} and run "
                    "audio packages verify --repair"
                ),
            )

        if not environment.has_interpreter:
            # `swift` has no interpreter and so nothing to sync, but it still holds a build
            # product and a Core ML package. It gets a registry entry anyway: reference
            # counting and purge treat every provisioned environment the same way, and an
            # untracked directory is one nothing can reclaim.
            if document["environments"].get(name, {}).get("state") == "ready":
                return False
            target.mkdir(parents=True, exist_ok=True)
            document["environments"][name] = {
                "state": "ready", "path": str(target), "python": None, "lock_sha256": None,
                "created_utc": _now(),
            }
            save_registry(document)
            return True

        entry = document["environments"].get(name, {})
        lock_digest = sha256_file(environment.lock)
        if entry.get("state") == "ready" and entry.get("lock_sha256") == lock_digest:
            return False

        # Intent first: a crash between here and the flip leaves a `creating` entry, which
        # reads as absent and as reclaimable rather than as a working environment.
        document["environments"][name] = {
            "state": "creating", "path": str(target), "python": environment.python,
            "lock_sha256": lock_digest, "created_utc": _now(),
        }
        save_registry(document)

        self.toolchain.create_environment(environment, target)
        document["environments"][name]["state"] = "ready"
        save_registry(document)
        return True

    # -- packages ----------------------------------------------------------------------

    def pull(self, selection: list[Package], *, repair: bool = False,
             stack: str | None = None) -> dict:
        """Materialize what is not already provisioned. Two asymmetries, both deliberate.

        **A `ready` package is skipped, not re-materialized.** Re-hashing a multi-gigabyte
        artifact, re-cloning and re-installing a checkout, or rebuilding the Swift product costs
        minutes and produces what is already there. Worse, the `pulling` entry that has to be
        written first would leave a working install downgraded if the pointless re-pull were
        interrupted — the crash-safety rule turned against a package nothing was wrong with.
        `--repair` is how a caller asks for the work anyway, and it *forces* re-materialization
        rather than trusting what is on disk, because a corrupt-but-present artifact is exactly
        the case it exists for.

        **A stack tolerates a toolchain-blocked package; a named one does not.** `--stack` is a
        superset guess, so an absent `swift` blocks `fluidaudio` and the rest of the stack still
        provisions, with the blocked package reported in `warnings` — which is what
        TRANSCRIBE_HAPPY_PATH.md §0 promises and what raising on the first blocked package broke,
        since `fluidaudio` sorts first and took the whole stack down with it. Naming a package is
        an instruction, so there the missing tool is still exit 3: silently skipping what a caller
        asked for by name is worse than refusing.
        """
        root_issue = managed_provisioning_root_issue()
        if root_issue is not None:
            raise ProvisioningError(
                "environment_drifted",
                root_issue,
                fix=(
                    f"Replace the redirected provisioning root {paths.root()} and run "
                    "audio packages verify --repair"
                ),
            )
        # Refuse a redirected root before even reading its registry.  Reading first
        # crosses the same ownership boundary as writing and can also surface attacker-
        # controlled JSON as a misleading registry error instead of the root failure.
        document = load_registry()
        # Read the cache once, before anything downloads. Everything after this point works
        # from that snapshot, so a revision fetched by this pull is never mistaken for one that
        # was already there.
        cached = self.fetcher.cached_revisions()
        pulled: list[dict] = []
        skipped: list[str] = []
        created: list[str] = []
        warnings: list[dict] = []
        blocked: list[tuple[Package, str]] = []

        for package in selection:
            if not repair and is_ready(document, package.id):
                # Nothing needs doing, so nothing is touched — in particular the entry is not
                # transitioned to `pulling`, which an interrupted no-op would leave behind.
                skipped.append(package.id)
                continue

            missing_tool = next((tool for tool in package.requires_tool
                                 if self.toolchain.which(tool) is None), None)
            if missing_tool is not None:
                if stack is None:
                    raise _toolchain_missing(package, missing_tool)
                blocked.append((package, missing_tool))
                continue

            if self.ensure_environment(package.environment, document):
                created.append(package.environment)

            previous = document["packages"].get(package.id, {})
            pre_existing = self._pre_existing_revisions(package, previous, cached)
            entry = {
                "state": "pulling", "environment": package.environment, "kind": package.kind,
                "source": package.source, "license_declared": package.license_declared,
                "license_reviewed": package.license_reviewed, "pulled_utc": _now(),
                # Decided before any bytes move, and carried across a retry: see
                # _pre_existing_revisions for why re-deciding would be wrong.
                "hub_revisions_pre_existing": sorted(pre_existing),
            }
            document["packages"][package.id] = entry
            save_registry(document)

            materialized = self._materialize(package, document, pre_existing, repair=repair)
            entry["materialized"] = materialized
            entry["state"] = "ready"
            document["packages"][package.id] = entry
            save_registry(document)

            receipt = {"package": package.id, "environment": package.environment,
                       "bytes": materialized.get("bytes")}
            for key in ("revision", "revisions", "digest_verified", "built", "product_runs",
                        "patches_applied"):
                if key in materialized:
                    receipt[key] = materialized[key]
            if materialized.get("hub_revisions_pre_existing"):
                receipt["hub_revisions_pre_existing"] = \
                    materialized["hub_revisions_pre_existing"]
                receipt["pre_existing_note"] = (
                    "already in the Hugging Face cache; not downloaded, and teardown here will "
                    "not delete it"
                )
            pulled.append(receipt)

        if blocked and not pulled and not skipped:
            # Nothing in the stack was provisionable, so there is no partial success to report
            # and the caller needs the exit code rather than a receipt of an empty pull.
            raise _toolchain_missing(*blocked[0])

        if blocked:
            tools = sorted({tool for _, tool in blocked})
            names = [package.id for package, _ in blocked]
            warnings.append({
                "code": "toolchain_missing", "blocking": True,
                "packages": names, "requires_tool": tools,
                "detail": (
                    f"{', '.join(names)} {'needs' if len(names) == 1 else 'need'} "
                    f"{', '.join(tools)}, which {'is' if len(tools) == 1 else 'are'} not on "
                    f"PATH, so {'it' if len(names) == 1 else 'they'} "
                    f"{'was' if len(names) == 1 else 'were'} not provisioned; the rest of "
                    f"{f'stack {stack}' if stack else 'the selection'} was. Install the "
                    f"toolchain and pull {'it' if len(names) == 1 else 'them'} by name."
                ),
            })

        # A blocked package provisioned nothing, so it carries no license claim and no bytes.
        # `pulled_known_bytes` says what this pull added, and a skipped package added none of it.
        blocked_ids = {package.id for package, _ in blocked}
        provisioned = [p for p in selection if p.id not in blocked_ids]
        unreviewed = sorted(p.id for p in provisioned if not p.license_reviewed)
        if unreviewed:
            warnings.append({
                "code": "license_unreviewed", "blocking": False,
                "packages": unreviewed,
                "detail": (
                    f"{', '.join(unreviewed)} "
                    f"{'reports' if len(unreviewed) == 1 else 'report'} a license their model "
                    "card declares but nobody has reviewed. A declared license is evidence "
                    "that one exists, not a redistribution clearance."
                ),
            })

        known, unsized = _selection_bytes(
            [p for p in provisioned if p.id not in set(skipped)], document)
        report = {
            "pulled": pulled,
            "skipped": skipped,
            "environments_created": sorted(set(created)),
            "root": str(paths.root()),
            "registry": str(paths.registry_path()),
            # This pull's packages only. `audio packages list` reports the cumulative total;
            # naming both `reclaimable` invited reading one as the other, and this figure
            # legitimately goes down between pulls.
            "pulled_known_bytes": known,
            "unsized_packages": unsized,
            "warnings": warnings,
        }
        if skipped:
            report["skipped_reason"] = (
                "already ready in the registry, so nothing was re-materialized and nothing "
                "counts toward pulled_known_bytes; audio packages pull --repair <package> "
                "re-materializes anyway, and audio packages verify is what checks them"
            )
        return report

    @staticmethod
    def _pre_existing_revisions(package: Package, previous: dict,
                                cached: set[str]) -> set[str]:
        """Which of this package's revisions the Hub cache held before this root wanted them.

        Two rules, and the second is the subtle one:

        - A revision already in the shared cache is not ours to delete later, whoever put it
          there — another tool, another provisioning root, or an earlier experiment.
        - **A retry must not re-decide.** `snapshot_download` publishes a snapshot directory as
          files land, so an interrupted 16 GiB pull leaves a revision that a later cache scan
          reports as present. Asking again would classify this root's own partly-finished
          download as somebody else's, and teardown would then refuse to reclaim 16 GiB it did
          in fact fetch. The first attempt's answer is the truthful one, so it is recorded in
          the `pulling` entry and reused.
        """
        if "hub_revisions_pre_existing" in previous:
            return set(previous["hub_revisions_pre_existing"])
        return {revision for revision in _source_revisions(package) if revision in cached}

    def _materialize(self, package: Package, document: dict, pre_existing: set[str], *,
                     repair: bool = False) -> dict:
        """Put the package on disk. `digest_verified` appears only where a digest was taken.

        One source kind pins a content hash — `url` — and it is the only one whose materialization
        can claim to have been verified against the manifest. The Hub kinds pin a *revision*; no
        `sha256` exists in the manifest to hash a snapshot against, so they record the revision
        and nothing more. They used to record `digest_verified: True` regardless, which made
        `verify` print `digest: "ok"` for a check no code performs.
        """
        kind = package.source["type"]
        if kind == "url":
            # The filename is manifest data, not derived: the shipped Silero backend resolves
            # this exact name, and a pull that invented one would leave two copies on disk and
            # re-download on first use. tests/test_environments.py ties the two together.
            target = paths.models_dir() / package.source["filename"]
            # `repair` needs no force here, and the digest is the reason: url_file re-hashes what
            # is on disk against the manifest pin and downloads again unless it matches, so a
            # match already is the strongest re-materialization available. Forcing the transfer
            # would spend the bytes to arrive at the same file.
            resolved = self.fetcher.url_file(package.source["url"], package.source["sha256"],
                                             target)
            _managed, location_issue = managed_url_artifact_path(package, resolved)
            if location_issue is not None:
                raise ProvisioningError(
                    "package_integrity_failed",
                    location_issue,
                    fix=f"audio packages pull --repair {package.id}",
                )
            return {"path": str(resolved), "bytes": _tree_bytes(resolved),
                    "digest_verified": True}

        if kind == "huggingface":
            revision = package.source["revision"]
            patterns = package.source.get("allow_patterns")
            if patterns is None:
                snapshot = self.fetcher.hf_snapshot(
                    package.source["repo"], revision, force=repair
                )
            else:
                snapshot = self.fetcher.hf_snapshot(
                    package.source["repo"],
                    revision,
                    force=repair,
                    allow_patterns=tuple(patterns),
                )
            snapshot_bytes = self._require_hub_snapshot(
                package,
                package.source["repo"],
                revision,
                snapshot,
                tuple(patterns or ()),
            )
            result = {
                "path": str(snapshot), "bytes": snapshot_bytes,
                "revision": revision,
                # Only what this pull fetched is ours to delete later, decided before the
                # download rather than after it — see _pre_existing_revisions.
                "hub_revisions": [] if revision in pre_existing else [revision],
                "hub_revisions_pre_existing": sorted(pre_existing),
            }
            result.update(self._checkout_and_install(package, repair=repair))
            return result

        if kind == "huggingface_multi":
            snapshots = {}
            total = 0
            ours: list[str] = []
            for repo in package.source["repos"]:
                patterns = repo.get("allow_patterns")
                snapshot = self.fetcher.hf_snapshot(
                    repo["repo"],
                    repo["revision"],
                    force=repair,
                    allow_patterns=tuple(patterns) if patterns is not None else None,
                )
                snapshot_bytes = self._require_hub_snapshot(
                    package,
                    repo["repo"],
                    repo["revision"],
                    snapshot,
                    tuple(patterns or ()),
                )
                snapshots[repo["repo"]] = str(snapshot)
                total += snapshot_bytes
                if repo["revision"] not in pre_existing:
                    ours.append(repo["revision"])
            result = {
                "paths": snapshots, "bytes": total,
                # Plural, because this package spans four repositories. A single `revision`
                # key would have to pick one of them, and the receipt promises the revisions
                # a pull materialized.
                "revisions": [repo["revision"] for repo in package.source["repos"]],
                "hub_revisions": ours, "hub_revisions_pre_existing": sorted(pre_existing),
            }
            result.update(self._checkout_and_install(package, repair=repair))
            return result

        if kind == "git+build":
            checkout = paths.checkout_dir(package.environment, package.id)
            # This method only runs for a non-ready package or an explicit repair. Reusing a
            # prior tree would let an interrupted pull's untracked build hook run before verify.
            _delete_managed(checkout)
            checkout.parent.mkdir(parents=True, exist_ok=True)
            self.toolchain.clone(package.source["repo"], package.source["commit"], checkout)
            applied, digests = materialize_checkout_patch(
                package, checkout, self.toolchain
            )
            try:
                state = self.toolchain.inspect_checkout(checkout)
            except ValueError as exc:
                raise ProvisioningError(
                    "checkout_integrity_failed",
                    f"could not inspect fresh checkout: {exc}",
                    package=package.id,
                    fix=f"audio packages pull --repair {package.id}",
                ) from exc
            _expected_patches, expected_names, _expected_digests = (
                checkout_patch_expectation(package)
            )
            if (
                state.head != package.source["commit"]
                or set(state.modified) != set(expected_names)
                or state.untracked
            ):
                raise ProvisioningError(
                    "checkout_integrity_failed",
                    f"{package.id} checkout was not exact before build",
                    package=package.id,
                    expected={
                        "head": package.source["commit"],
                        "modified": sorted(expected_names),
                        "untracked": [],
                    },
                    actual={
                        "head": state.head,
                        "modified": list(state.modified),
                        "untracked": list(state.untracked),
                    },
                    fix=f"audio packages pull --repair {package.id}",
                )
            self.toolchain.swift_build(checkout)
            product = package.source["product"]
            runs = self.toolchain.swift_product_runs(checkout, product)
            if not runs:
                # A build that produces an executable nobody can launch is not a provisioned
                # package, and this used to be recorded and then ignored: `pull` returned exit 0
                # with an empty `warnings`, `verify` reported `failed: []`, and the environment
                # read `ok`, while the one thing the package exists to do was impossible. The
                # entry stays `pulling`, so `list` and `run` both report it absent.
                raise ProvisioningError(
                    "package_build_unusable",
                    f"{package.id} built, but its product {product!r} does not run, so nothing "
                    f"in the {package.environment} environment can use it",
                    package=package.id, product=product, built=True,
                    fix=f"audio packages pull --repair {package.id}",
                )
            candidates = built_product_candidates(checkout, product)
            if len(candidates) != 1:
                raise ProvisioningError(
                    "package_build_unusable",
                    f"{package.id} built, but expected one contained executable product "
                    f"{product!r} and found {len(candidates)}",
                    package=package.id, product=product, built=True,
                    fix=f"audio packages pull --repair {package.id}",
                )
            executable = candidates[0]
            product_path = executable.relative_to(checkout.resolve(strict=True)).as_posix()
            return {"path": str(checkout), "bytes": _tree_bytes(checkout / ".build"),
                    "revision": package.source["commit"], "built": True,
                    "product_runs": runs, "product_path": product_path,
                    "product_sha256": sha256_file(executable),
                    "patches_applied": applied,
                    "patched_file_digests": digests}

        raise ManifestError(f"{package.id}: unsupported source type {kind!r}")

    @staticmethod
    def _require_hub_snapshot(
        package: Package,
        repository: str,
        revision: str,
        snapshot: Path,
        patterns: tuple[str, ...],
    ) -> int:
        """Bind and measure one download before pull performs any subsequent work."""
        try:
            snapshot_index = _hub_snapshot_index()
        except Exception as exc:  # noqa: BLE001 - no index means no trusted materialization
            issues = [f"Hugging Face cache identity cannot be inspected: {exc}"]
        else:
            issues, byte_count = _inspect_hub_snapshot(
                repository, revision, snapshot, patterns, snapshot_index
            )
        if issues:
            raise ProvisioningError(
                "package_integrity_failed",
                "; ".join(issues),
                package=package.id,
                fix=f"audio packages pull --repair {package.id}",
            )
        assert byte_count is not None
        return byte_count

    def _checkout_and_install(self, package: Package, *, repair: bool = False) -> dict:
        """Pinned source checkout, patch, and a --no-deps install into the environment."""
        if package.checkout is None:
            return {}
        checkout = paths.checkout_dir(package.environment, package.id)
        # A ready package is skipped before this method. Every invocation therefore represents
        # a fresh materialization or explicit repair and starts from an absent checkout; otherwise
        # an ordinary untracked setup hook could execute during install before later verification.
        _delete_managed(checkout)
        checkout.parent.mkdir(parents=True, exist_ok=True)
        resolved_commit = package.checkout.get(
            "resolved_commit", package.checkout["commit"])
        self.toolchain.clone(package.checkout["repo"], resolved_commit, checkout)

        applied, digests = materialize_checkout_patch(
            package, checkout, self.toolchain
        )
        _expected_patches, expected_names, _expected_digests = (
            checkout_patch_expectation(package)
        )
        try:
            state = self.toolchain.inspect_checkout(checkout)
        except ValueError as exc:
            raise ProvisioningError(
                "checkout_integrity_failed", f"could not inspect fresh checkout: {exc}",
                package=package.id, fix=f"audio packages pull --repair {package.id}",
            ) from exc
        if state.head != resolved_commit or set(state.modified) != set(expected_names) \
                or state.untracked:
            raise ProvisioningError(
                "checkout_integrity_failed",
                f"{package.id} checkout was not exact before install",
                package=package.id,
                expected={
                    "head": resolved_commit,
                    "modified": sorted(expected_names),
                    "untracked": [],
                },
                actual={
                    "head": state.head,
                    "modified": list(state.modified),
                    "untracked": list(state.untracked),
                },
                fix=f"audio packages pull --repair {package.id}",
            )

        self.toolchain.install_checkout(paths.env_python(package.environment), checkout)
        # Stage interpreters use ``-B`` so they never recreate ignored bytecode.  Cleaning
        # anything the wheel build left in the source tree makes a successful pull start
        # from the same inspectable state that run preflight requires.
        self.toolchain.clean_ignored_checkout(checkout)
        return {"checkout": str(checkout), "checkout_commit": resolved_commit,
                "patches_applied": applied, "patched_file_digests": digests}

    # -- verify ------------------------------------------------------------------------

    def verify(self, *, repair: bool = False) -> dict:
        document = load_registry()
        catalog = packages()
        verified: list[dict] = []
        failed: list[dict] = []
        environment_states: dict[str, str] = {}
        built_runtime_probes: dict[str, bool] = {}
        invalid_environment_roots: set[str] = set()
        ready_packages_by_environment: dict[str, list[str]] = {}
        for identifier, package_entry in document["packages"].items():
            package = catalog.get(identifier)
            if package is not None and package_entry.get("state") == "ready":
                ready_packages_by_environment.setdefault(package.environment, []).append(
                    identifier
                )

        for name, environment in environments().items():
            if not environment.provisioned:
                continue
            entry = document["environments"].get(name)
            if entry is None or entry.get("state") != "ready":
                environment_states[name] = "absent"
                dependents = sorted(ready_packages_by_environment.get(name, ()))
                if dependents:
                    invalid_environment_roots.add(name)
                    actual_state = entry.get("state") if isinstance(entry, dict) else None
                    failed.append({
                        "environment": name,
                        "code": "environment_not_ready",
                        "detail": (
                            f"registry state is {actual_state or 'absent'!r} while ready "
                            f"package(s) depend on it: {', '.join(dependents)}"
                        ),
                        "packages": dependents,
                        "fix": (
                            "audio packages pull --repair " + " ".join(dependents)
                        ),
                    })
                continue
            environment_path, environment_path_issue = managed_environment_path(name)
            if environment_path_issue is not None:
                invalid_environment_roots.add(name)
                environment_states[name] = "drifted"
                failed.append({
                    "environment": name,
                    "code": "environment_drifted",
                    "detail": environment_path_issue,
                    "examples": {
                        "environment_root": {
                            "locked": str(paths.env_dir(name)),
                            "installed": str(environment_path),
                        }
                    },
                    "fix": (
                        f"Replace the redirected environment path {paths.env_dir(name)} and "
                        "run audio packages verify --repair"
                    ),
                })
                continue
            # Swift is a provisioning dependency until its product exists.  Once the executable
            # is ready, runtime and verify launch it directly, so removing Swift from PATH must
            # not turn a still-runnable environment into `blocked`.
            blocked_by = [tool for tool in environment.requires_tool
                          if self.toolchain.which(tool) is None]
            if blocked_by:
                built_runtime_probes[name] = _environment_built_runtime_runs(
                    name, document, self.toolchain,
                )
                if not built_runtime_probes[name]:
                    environment_states[name] = "blocked"
                    continue
            if not environment.has_interpreter:
                # No interpreter means no lock to compare against. Its packages carry the
                # checks that apply — that the product builds and runs — so reporting `ok`
                # here says the directory exists, and nothing more.
                environment_states[name] = "ok"
                continue
            expected = _locked_versions(environment)
            frozen = self.toolchain.frozen_packages(paths.env_python(name))
            required_checkouts = _managed_checkout_requirements(document, name)
            drift = _environment_drift(expected, frozen, required_checkouts)
            if drift and repair:
                ready_checkouts: list[tuple[Package, Path]] = []
                checkouts_are_safe = True
                for identifier, package_entry in sorted(document["packages"].items()):
                    package = catalog.get(identifier)
                    if (
                        package is None
                        or package.environment != name
                        or package.checkout is None
                        or package_entry.get("state") != "ready"
                    ):
                        continue
                    materialized = package_entry.get("materialized", {})
                    if not isinstance(materialized, dict):
                        checkouts_are_safe = False
                        continue
                    issues = _checkout_integrity_issues(
                        package, materialized, self.toolchain
                    )
                    checkout, checkout_issue = managed_checkout_path(
                        package, materialized.get("checkout")
                    )
                    if issues or checkout_issue is not None or checkout is None:
                        checkouts_are_safe = False
                        continue
                    ready_checkouts.append((package, checkout))
                if checkouts_are_safe:
                    lock_digest = sha256_file(environment.lock)
                    document["environments"][name] = {
                        "state": "creating",
                        "path": str(paths.env_dir(name)),
                        "python": environment.python,
                        "lock_sha256": lock_digest,
                        "created_utc": _now(),
                    }
                    save_registry(document)
                    self.toolchain.create_environment(environment, paths.env_dir(name))
                    for _package, checkout in ready_checkouts:
                        self.toolchain.install_checkout(
                            paths.env_python(name), checkout
                        )
                        self.toolchain.clean_ignored_checkout(checkout)
                    document["environments"][name]["state"] = "ready"
                    save_registry(document)
                    frozen = self.toolchain.frozen_packages(paths.env_python(name))
                    drift = _environment_drift(
                        expected, frozen, required_checkouts
                    )
            if drift:
                environment_states[name] = "drifted"
                failed.append({
                    "environment": name, "code": "environment_drifted",
                    "detail": f"{len(drift)} package(s) differ from {environment.lock.name}",
                    "examples": {n: {"locked": drift[n][0], "installed": drift[n][1]}
                                 for n in sorted(drift)[:5]},
                    "fix": "audio packages verify --repair",
                })
            else:
                environment_states[name] = "ok"

        for identifier, entry in sorted(document["packages"].items()):
            if entry.get("state") != "ready":
                failed.append({"package": identifier, "code": "package_not_ready",
                               "detail": f"state is {entry.get('state')!r}",
                               "fix": f"audio packages pull --repair {identifier}"})
                continue
            package = catalog.get(identifier)
            if package is None:
                failed.append({
                    "package": identifier,
                    "code": "package_unknown",
                    "detail": "ready registry entry is not present in the installed manifest",
                    "fix": f"Inspect {paths.registry_path()} and remove the stale entry",
                })
                continue
            # The environment-root verdict owns every path below it.  Once that root is
            # redirected, do not inspect a checkout or launch a product reached through it;
            # the environment failure above is the complete, safe diagnosis.
            if package.environment in invalid_environment_roots:
                continue
            record: dict = {"package": identifier}
            materialized = entry.get("materialized", {})
            if not isinstance(materialized, dict):
                failed.append({
                    "package": identifier,
                    "code": "package_integrity_failed",
                    "detail": "materialized receipt is not an object",
                    "fix": f"audio packages pull --repair {identifier}",
                })
                continue
            # `digest: "ok"` is reserved for the one kind that has something to hash against.
            if package.source["type"] == "url":
                location, location_issue = managed_url_artifact_path(
                    package, materialized.get("path")
                )
                digest_matches = False
                if location_issue is None and location is not None:
                    try:
                        digest_matches = (
                            sha256_file(location) == package.source["sha256"]
                        )
                    except OSError as exc:
                        location_issue = f"could not hash managed artifact {location}: {exc}"
                if digest_matches:
                    record["digest"] = "ok"
                else:
                    failed.append({
                        "package": identifier, "code": "package_integrity_failed",
                        "detail": location_issue or (
                            f"{location} is missing or its digest changed"
                        ),
                        "fix": f"audio packages pull --repair {identifier}",
                    })
                    continue
            elif package.source["type"] in {
                "huggingface", "huggingface_multi",
            }:
                issues = hub_materialization_issues(package, materialized)
                if issues:
                    failed.append({
                        "package": identifier, "code": "package_integrity_failed",
                        "detail": "; ".join(issues),
                        "fix": f"audio packages pull --repair {identifier}",
                    })
                    continue
                # The receipt is not the pin.  It is mutable local history and may predate a
                # manifest correction; only the manifest revisions can be republished as the
                # revisions this verification actually required above.
                record.update(_source_revision_report(package))
            elif package.source["type"] == "git+build":
                checkout_value = materialized.get("path")
                checkout, location_issue = managed_checkout_path(package, checkout_value)
                product = str(package.source["product"])
                issues: list[str] = []
                if location_issue is not None:
                    issues.append(location_issue)
                elif checkout is None or not checkout.is_dir():
                    issues.append(f"built checkout is not a directory: {checkout}")
                else:
                    try:
                        state = self.toolchain.inspect_checkout(checkout)
                    except ValueError as exc:
                        issues.append(str(exc))
                    else:
                        if state.head != package.source["commit"]:
                            issues.append(
                                f"checkout HEAD is {state.head!r}, expected exact commit "
                                f"{package.source['commit']!r}"
                            )
                        try:
                            expected_patches, expected_names, expected_digests = (
                                checkout_patch_expectation(package)
                            )
                        except (OSError, ValueError) as exc:
                            issues.append(f"shipped checkout patch cannot be read: {exc}")
                            expected_patches, expected_names, expected_digests = (), (), {}
                        if set(state.modified) != set(expected_names):
                            issues.append(
                                f"built checkout tracked changes are "
                                f"{list(state.modified)!r}, expected "
                                f"{sorted(expected_names)!r}"
                            )
                        applied = materialized.get("patches_applied", [])
                        if applied != list(expected_patches):
                            issues.append(
                                f"recorded patches are {applied!r}, expected exactly "
                                f"{list(expected_patches)!r}"
                            )
                        recorded_digests = materialized.get(
                            "patched_file_digests", {}
                        )
                        if recorded_digests != expected_digests:
                            issues.append(
                                "recorded patched-file digests differ from manifest"
                            )
                        changed = [
                            name for name, digest in expected_digests.items()
                            if not checkout_file_matches(
                                checkout, name, digest, self.toolchain.file_digest
                            )
                        ]
                        if changed:
                            issues.append(
                                f"live patched-file hashes changed for {sorted(changed)!r}"
                            )
                    candidates = built_product_candidates(checkout, product)
                    if len(candidates) != 1:
                        issues.append(
                            f"expected one executable built product {product!r}, found "
                            f"{len(candidates)}"
                        )
                    else:
                        _executable, product_issue = validated_built_product(
                            checkout, product, materialized
                        )
                        if product_issue is not None:
                            issues.append(product_issue)
                if issues:
                    failed.append({
                        "package": identifier, "code": "package_integrity_failed",
                        "detail": "; ".join(issues),
                        "fix": f"audio packages pull --repair {identifier}",
                    })
                    continue
                if package.environment in built_runtime_probes:
                    product_runs = built_runtime_probes[package.environment]
                else:
                    try:
                        product_runs = self.toolchain.built_product_runs(candidates[0])
                    except OSError:
                        product_runs = False
                if not product_runs:
                    failed.append({
                        "package": identifier, "code": "package_build_unusable",
                        "detail": f"built product {product!r} does not run",
                        "fix": f"audio packages pull --repair {identifier}",
                    })
                    continue
                record["product_runs"] = True
                record["product_digest"] = "ok"
                record["patches_applied"] = list(expected_patches)
            else:
                locations = [Path(p) for p in (
                    [materialized["path"]] if materialized.get("path")
                    else list((materialized.get("paths") or {}).values()))]
                gone = [str(location) for location in locations if not location.exists()]
                if gone:
                    # The shared-cache consequence: another root's purge, or a manual cache
                    # clear, can take weights out from under a root that still calls them
                    # ready. Better an exit 3 with a fix than a stack that fails mid-run.
                    failed.append({
                        "package": identifier, "code": "package_integrity_failed",
                        "detail": f"materialized path(s) no longer exist: {', '.join(gone)}",
                        "fix": f"audio packages pull --repair {identifier}",
                    })
                    continue

            if package.checkout is not None:
                issues = _checkout_integrity_issues(package, materialized, self.toolchain)
                if issues:
                    failed.append({
                        "package": identifier, "code": "package_integrity_failed",
                        "detail": "; ".join(issues),
                        "fix": f"audio packages pull --repair {identifier}",
                    })
                    continue
                record["patches_applied"] = materialized.get("patches_applied", [])
            verified.append(record)

        report: dict = {"verified": verified, "environments": environment_states,
                        "failed": failed}
        report.update(self._verify_mlx_guard(document))
        return report

    def _verify_mlx_guard(self, document: dict) -> dict:
        """The pinned private decode path, checked without loading a checkpoint."""
        environment = environments()["mlx"]
        guards = {guard["kind"]: guard for guard in environment.guards}
        expected = guards["source_hash"]["sha256"]
        report: dict = {"mlx_audio_private_api_expected_source_hash": expected}
        if document["environments"].get("mlx", {}).get("state") != "ready":
            report["mlx_audio_private_api_source_hash"] = None
            report["mlx_audio_private_api_matches_expected"] = None
            return report
        _managed, environment_issue = managed_environment_path("mlx")
        if environment_issue is not None:
            report["mlx_audio_private_api_source_hash"] = None
            report["mlx_audio_private_api_matches_expected"] = False
            report["mlx_audio_private_api_error"] = environment_issue
            return report

        target = guards["source_hash"]["target"]
        signature = guards["signature"]
        probe = (
            "import hashlib, inspect, json\n"
            f"mod = __import__({_module_of(signature['target'])!r}, fromlist=['x'])\n"
            "path = inspect.getfile(mod)\n"
            f"cls = getattr(mod, {_class_of(signature['target'])!r})\n"
            f"method = getattr(cls, {_method_of(signature['target'])!r})\n"
            "print(json.dumps({'sha256': hashlib.sha256(open(path,'rb').read()).hexdigest(),"
            " 'params': sorted(inspect.signature(method).parameters)}))\n"
        )
        result = self.toolchain.run([str(paths.env_python("mlx")), "-c", probe])
        if result.returncode != 0:
            report["mlx_audio_private_api_source_hash"] = None
            report["mlx_audio_private_api_matches_expected"] = False
            report["mlx_audio_private_api_error"] = result.stderr.strip()[-400:]
            return report
        lines = result.stdout.strip().splitlines()
        try:
            observed = json.loads(lines[-1]) if lines else {}
        except json.JSONDecodeError:
            observed = {}
        if "sha256" not in observed:
            report["mlx_audio_private_api_source_hash"] = None
            report["mlx_audio_private_api_matches_expected"] = None
            report["mlx_audio_private_api_error"] = (
                "the environment's interpreter produced no verdict; nothing is claimed"
            )
            return report
        report["mlx_audio_private_api_source_hash"] = observed["sha256"]
        report["mlx_audio_private_api_matches_expected"] = observed["sha256"] == expected
        report["mlx_audio_private_api_signature_ok"] = set(
            signature["required_parameters"]).issubset(observed["params"])
        report["mlx_audio_private_api_target"] = target
        return report

    # -- teardown ----------------------------------------------------------------------

    def remove(self, package_ids: list[str]) -> dict:
        """Remove named packages. Every name is resolved before anything is deleted.

        The ordering is the point. Deleting inside the same loop that raised on an unknown name,
        with one `save_registry` after it, meant `remove vibevoice-asr-7b firered-asr2` discarded
        17 GiB and then rolled the registry *back* — leaving an entry that still read `ready` for
        a package whose bytes were gone. Nothing downstream notices, because `missing_packages`
        keys on `state`, so the caller finds out at model load. That is the mirror image of what
        the `pulling` state exists to prevent: a pull that dies is honest about being incomplete,
        and a teardown that died was not. A caller naming several packages already assumes
        all-or-nothing, so validate the whole list first.

        Two consequences, both intended. A repeated name removes once and is reported once. And
        each entry is dropped and saved as soon as its own files are gone rather than in one write
        at the end, so no later failure in the teardown can restore a claim to bytes that no
        longer exist.
        """
        document = load_registry()
        targets: list[str] = []
        for identifier in package_ids:
            if identifier not in document["packages"]:
                raise ProvisioningError(
                    "package_not_provisioned", f"{identifier} is not in the registry",
                    exit_code=2, package=identifier, fix="audio packages list",
                )
            if identifier not in targets:
                targets.append(identifier)

        removed: list[str] = []
        hub_revisions: list[str] = []
        retained: list[str] = []
        local_freed = 0
        catalog = packages()

        for identifier in targets:
            materialized = document["packages"][identifier].get("materialized", {})
            materialized = materialized if isinstance(materialized, dict) else {}
            owned, kept_revisions = _teardown_revisions(
                catalog.get(identifier), materialized
            )
            hub_revisions.extend(owned)
            retained.extend(kept_revisions)
            for location in _managed_package_locations(catalog.get(identifier)):
                local_freed += _delete_managed(location)
            document["packages"].pop(identifier)
            removed.append(identifier)
            save_registry(document)

        # After the entries are gone, so a cache that fails here costs reclaimable space in a
        # shared cache rather than leaving a package whose local bytes are already deleted
        # reading as ready. There is no ordering that keeps that entry honest.
        deleted, hub_freed = self.fetcher.delete_hub_revisions(hub_revisions)
        kept, dropped, environment_freed = self._collect_environments(document)
        local_freed += environment_freed
        save_registry(document)
        report = {
            "removed": removed,
            "environments_removed": dropped,
            "environments_kept": kept,
            "hub_revisions_deleted": deleted,
            "hub_revisions_not_found": sorted(set(hub_revisions) - set(deleted)),
            "hub_revisions_retained": sorted(set(retained)),
            "hub_cache_note": HUB_CACHE_NOTE,
            "reclaimed_bytes": hub_freed + local_freed,
        }
        if retained:
            report["hub_revisions_retained_reason"] = (
                "not owned by this root under the current package manifest, so they are not "
                "this root's to delete"
            )
        if dropped:
            report["environments_removed_reason"] = (
                f"no other provisioned package targets {', '.join(dropped)}")
        if kept:
            report["environments_kept_reason"] = "; ".join(
                f"{', '.join(sorted(users))} still "
                f"{'needs' if len(users) == 1 else 'need'} {name}"
                for name, users in sorted(_users_by_environment(document).items())
                if name in kept)
        return report

    def _collect_environments(self, document: dict) -> tuple[list[str], list[str], int]:
        """Reference counting, derived from the package table each time it is asked."""
        users = _users_by_environment(document)
        kept, dropped, freed = [], [], 0
        known = environments()
        for name in sorted(document["environments"]):
            if users.get(name):
                kept.append(name)
                continue
            if name in known:
                freed += _delete_managed(paths.env_dir(name))
            document["environments"].pop(name)
            dropped.append(name)
        return kept, dropped, freed

    def purge(self, *, dry_run: bool) -> dict:
        document = load_registry()
        package_ids = sorted(document["packages"])
        environment_names = sorted(document["environments"])
        known, unsized = _selection_bytes(
            [packages()[i] for i in package_ids if i in packages()], document)

        if dry_run:
            deletable, keeping = [], []
            catalog = packages()
            for identifier in package_ids:
                materialized = document["packages"][identifier].get("materialized", {})
                materialized = materialized if isinstance(materialized, dict) else {}
                owned, retained_revisions = _teardown_revisions(
                    catalog.get(identifier), materialized
                )
                deletable.extend(owned)
                keeping.extend(retained_revisions)
            return {
                "would_remove": {"packages": package_ids, "environments": environment_names,
                                 "root": str(paths.root()),
                                 "hub_revisions": sorted(set(deletable))},
                "would_keep": {"hub_revisions": sorted(set(keeping))},
                "hub_cache_note": HUB_CACHE_NOTE,
                "reclaimable_known_bytes": known,
                "reclaimable_note": (
                    "projected from manifest-bounded registry receipts and package sizes; "
                    "it counts deletion-eligible Hub weights and recorded local package "
                    "artifacts, but excludes retained revisions and environment bytes"
                ),
                "unsized_packages": unsized,
                "untouched": UNTOUCHED,
            }

        hub_revisions: list[str] = []
        retained: list[str] = []
        local_freed = 0
        catalog = packages()
        # `purge` cannot take a name that is not in the registry — it reads the list *from* the
        # registry — so it never had `remove`'s validation defect. It shared the narrower half:
        # every local file was deleted and the registry was cleared in one write afterwards, so
        # anything that raised in between left every package reading as `ready` with nothing
        # behind it. Same rule as `remove`, then: an entry goes as soon as its own bytes do.
        for identifier in package_ids:
            materialized = document["packages"][identifier].get("materialized", {})
            materialized = materialized if isinstance(materialized, dict) else {}
            owned, kept_revisions = _teardown_revisions(
                catalog.get(identifier), materialized
            )
            hub_revisions.extend(owned)
            retained.extend(kept_revisions)
            for location in _managed_package_locations(catalog.get(identifier)):
                local_freed += _delete_managed(location)
            document["packages"].pop(identifier)
            save_registry(document)
        known_environments = environments()
        for name in environment_names:
            if name in known_environments:
                local_freed += _delete_managed(paths.env_dir(name))
            document["environments"].pop(name, None)
            save_registry(document)
        deleted, hub_freed = self.fetcher.delete_hub_revisions(hub_revisions)
        return {
            "removed": {"packages": package_ids, "environments": environment_names},
            "hub_revisions_deleted": deleted,
            "hub_revisions_not_found": sorted(set(hub_revisions) - set(deleted)),
            "hub_revisions_retained": sorted(set(retained)),
            "hub_cache_note": HUB_CACHE_NOTE,
            "reclaimed_bytes": hub_freed + local_freed,
            "unsized_packages": unsized,
            "untouched": UNTOUCHED,
        }


# --------------------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------------------


def doctor(toolchain: Toolchain | None = None) -> dict:
    """Everything a reader needs before deciding whether a failure is theirs or the tool's."""
    from . import __version__

    toolchain = toolchain or Toolchain()
    document = load_registry()
    tools = {}
    for tool in ("ffmpeg", "ffprobe", "swift", "uv", "git"):
        location = toolchain.which(tool)
        tools[tool] = {"present": location is not None, "path": location}
    tools["huggingface_hub"] = {"present": _module_available("huggingface_hub"), "path": None}

    root = paths.root()
    usage = shutil.disk_usage(root if root.exists() else Path.home())
    return {
        "tool": {"version": __version__, "path": sys.argv[0],
                 "python": platform.python_version()},
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine()},
        "memory": _memory(),
        "disk": {"total_bytes": usage.total, "free_bytes": usage.free},
        "tools": tools,
        "root": str(root),
        "root_exists": root.exists(),
        "registry": str(paths.registry_path()),
        "environments": {
            name: {
                "state": document["environments"].get(name, {}).get("state", "absent"),
                "python": environment.python,
                "requires_tool": list(environment.requires_tool),
                "blocked_by_missing_tool": [
                    tool for tool in environment.requires_tool
                    if not tools[tool]["present"]
                    and not _environment_has_built_runtime(name, document)
                ],
                "provisional": environment.provisional,
            }
            for name, environment in environments().items() if environment.provisioned
        },
        "packages": {
            identifier: document["packages"].get(identifier, {}).get("state", "absent")
            for identifier in sorted(packages())
        },
        "note": (
            "Swift is required to build or repair FluidAudio; a ready built product runs "
            "directly without Swift. Missing provisioning tools are reported rather than fatal."
        ),
    }


def _memory() -> dict:
    """Total and available memory, or nulls where the platform does not report them."""
    total = available = None
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    if sys.platform == "darwin":
        try:
            result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10)
            page = 4096
            free = inactive = 0
            for line in result.stdout.splitlines():
                if "page size of" in line:
                    page = int(line.split("page size of")[1].split()[0])
                elif line.startswith("Pages free:"):
                    free = int(line.split(":")[1].strip().rstrip("."))
                elif line.startswith("Pages inactive:"):
                    inactive = int(line.split(":")[1].strip().rstrip("."))
            available = (free + inactive) * page
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return {"total_bytes": total, "available_bytes": available,
            "note": "Host-wide counters; not process-attributable and not summable with "
                    "per-stage peaks."}


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _module_available(name: str) -> bool:
    from importlib.util import find_spec

    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _module_of(dotted: str) -> str:
    return dotted.rsplit(".", 2)[0]


def _class_of(dotted: str) -> str:
    return dotted.rsplit(".", 2)[1]


def _method_of(dotted: str) -> str:
    return dotted.rsplit(".", 1)[1]


def _locked_versions(environment: Environment) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in environment.lock.read_text().splitlines():
        if line.startswith((" ", "#")) or "==" not in line:
            continue
        name, _, rest = line.partition("==")
        versions[_distribution_name(name)] = rest.split()[0].strip(" \\")
    return versions


def _distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip().lower())


def _direct_file_install_path(specification: str) -> Path | None:
    if not specification.startswith("@ "):
        return None
    try:
        parsed = urllib.parse.urlsplit(specification[2:].strip())
    except ValueError:
        return None
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        return None
    return Path(urllib.parse.unquote(parsed.path))


def _managed_checkout_requirements(
    document: Mapping[str, Any],
    environment_name: str,
    package_ids: set[str] | None = None,
) -> dict[str, Path]:
    """Required direct installs, keyed by manifest-pinned distribution name."""
    catalog = packages()
    entries = document.get("packages", {})
    if not isinstance(entries, Mapping):
        return {}
    required: dict[str, Path] = {}
    for identifier, entry in entries.items():
        if package_ids is not None and identifier not in package_ids:
            continue
        package = catalog.get(identifier)
        if (
            not isinstance(entry, Mapping)
            or entry.get("state") != "ready"
            or package is None
            or package.environment != environment_name
            or package.checkout is None
        ):
            continue
        distribution = _distribution_name(str(package.checkout["distribution"]))
        required[distribution] = Path(os.path.abspath(
            paths.checkout_dir(package.environment, identifier)
        ))
    return required


def _environment_drift(
    expected: dict[str, str],
    frozen: dict[str, str],
    required_checkouts: Mapping[str, Path],
) -> dict[str, tuple[str | None, str | None]]:
    comparable = dict(frozen)
    direct_drift = _checkout_install_drift(frozen, required_checkouts)
    for name in required_checkouts:
        comparable.pop(name, None)
    locked_drift = {
        name: (expected.get(name), comparable.get(name))
        for name in expected.keys() | comparable.keys()
        if expected.get(name) != comparable.get(name)
    }
    return {**locked_drift, **direct_drift}


def _checkout_install_drift(
    frozen: Mapping[str, str],
    required_checkouts: Mapping[str, Path],
) -> dict[str, tuple[str | None, str | None]]:
    drift: dict[str, tuple[str | None, str | None]] = {}
    for name, required_path in required_checkouts.items():
        installed = frozen.get(name)
        direct_path = (
            _direct_file_install_path(installed) if installed is not None else None
        )
        if direct_path is None or Path(os.path.abspath(direct_path)) != required_path:
            drift[name] = (f"@ {required_path.as_uri()}", installed)
    return drift


def _patch_touched_names(patch: Path) -> tuple[str, ...]:
    """Repository-relative files touched by one shipped unified diff."""
    touched: list[str] = []
    for line in patch.read_text(errors="replace").splitlines():
        if line.startswith("+++ ") and not line.endswith("/dev/null"):
            target = line[4:].strip()
            if target.startswith("b/"):
                target = target[2:]
            touched.append(target)
    return tuple(touched)


def _patched_files(patch: Path, checkout: Path) -> list[Path]:
    """Files a unified diff touches, so verify can detect a reverted patch."""
    return [checkout / name for name in _patch_touched_names(patch)]


def checkout_patch_expectation(
    package: Package,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, str]]:
    """Exact receipt, tracked-file set, and post-patch hashes owned by the manifest.

    The registry is mutable history, not an integrity root.  The manifest binds the
    exact post-patch bytes independently of Git diff presentation and receipt values.
    """
    specification = (
        package.source
        if package.source.get("type") == "git+build"
        else package.checkout
    )
    if specification is None:
        return (), (), {}
    expected = dict(specification.get("patched_file_sha256", {}))
    patch_name = specification.get("patch")
    if not patch_name:
        return (), (), expected
    patch = ENVIRONMENTS_DIR / str(patch_name)
    names = _patch_touched_names(patch)
    if set(expected) != set(names):
        raise ValueError(
            f"manifest patched_file_sha256 paths {sorted(expected)!r} do not equal "
            f"shipped patch targets {sorted(names)!r}"
        )
    return (Path(str(patch_name)).name,), names, expected


def materialize_checkout_patch(
    package: Package,
    checkout: Path,
    toolchain: Toolchain,
) -> tuple[list[str], dict[str, str]]:
    """Apply and prove the manifest-owned patch before install or build executes."""
    expected_patches, expected_names, expected_digests = (
        checkout_patch_expectation(package)
    )
    applied: list[str] = []
    digests: dict[str, str] = {}
    if expected_patches:
        specification = (
            package.source
            if package.source.get("type") == "git+build"
            else package.checkout
        )
        assert specification is not None
        patch_name = str(specification["patch"])
        patch = ENVIRONMENTS_DIR / patch_name
        if not patch.is_file():
            raise ProvisioningError(
                "patch_missing", f"{patch} is not in the installed wheel",
                patch=patch_name, package=package.id,
            )
        toolchain.apply_patch(checkout, patch)
        applied.append(Path(patch_name).name)
        for touched in _patched_files(patch, checkout):
            if touched.is_file():
                digests[str(touched.relative_to(checkout))] = (
                    toolchain.file_digest(touched)
                )
    if (
        applied != list(expected_patches)
        or set(digests) != set(expected_names)
        or digests != expected_digests
    ):
        raise ProvisioningError(
            "patch_integrity_failed",
            f"{package.id} did not materialize the manifest-pinned patched file hashes",
            package=package.id,
            expected=expected_digests,
            actual=digests,
            fix=f"audio packages pull --repair {package.id}",
        )
    return applied, digests


def managed_checkout_path(
    package: Package, value: object,
) -> tuple[Path | None, str | None]:
    """Resolve only the manifest-derived checkout, rejecting symlink/parent escapes."""
    expected = paths.checkout_dir(package.environment, package.id)
    candidate = Path(str(value)) if value else None
    if candidate is None:
        return None, "checkout path is absent"
    _environment_path, environment_issue = managed_environment_path(package.environment)
    if environment_issue is not None:
        return candidate, f"package environment is not managed: {environment_issue}"
    if Path(os.path.abspath(candidate)) != Path(os.path.abspath(expected)):
        return candidate, f"checkout path {candidate} is not managed path {expected}"
    if candidate.is_symlink():
        return candidate, f"managed checkout path is a symlink: {candidate}"
    try:
        resolved = candidate.resolve(strict=True)
        root = paths.root().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return candidate, f"managed checkout path cannot be resolved: {exc}"
    if not resolved.is_relative_to(root):
        return candidate, f"managed checkout resolves outside provisioning root: {resolved}"
    return candidate, None


def managed_provisioning_root_issue(*, create: bool = False) -> str | None:
    """Require the configured provisioning-root leaf to be a real directory."""

    root = paths.root()
    try:
        state = os.stat(root, follow_symlinks=False)
    except FileNotFoundError:
        if not create:
            return None
        try:
            root.mkdir(parents=True, exist_ok=True)
            state = os.stat(root, follow_symlinks=False)
        except (OSError, RuntimeError) as exc:
            return f"provisioning root cannot be created safely: {root}: {exc}"
    except (OSError, RuntimeError) as exc:
        return f"provisioning root cannot be inspected safely: {root}: {exc}"
    if stat.S_ISLNK(state.st_mode):
        return f"provisioning root is a symlink: {root}"
    if not stat.S_ISDIR(state.st_mode):
        return f"provisioning root is not a directory: {root}"
    return None


def managed_environment_path(name: str) -> tuple[Path, str | None]:
    """Require the manifest-derived environment root without following an inner symlink."""
    expected = paths.env_dir(name)
    root_issue = managed_provisioning_root_issue()
    if root_issue is not None:
        return expected, root_issue
    if expected.parent.is_symlink():
        return expected, f"managed environment parent is a symlink: {expected.parent}"
    if expected.is_symlink():
        return expected, f"managed environment path is a symlink: {expected}"
    try:
        resolved = expected.resolve(strict=True)
        root = paths.root().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return expected, f"managed environment path cannot be resolved: {exc}"
    if not expected.is_dir() or not resolved.is_relative_to(root):
        return expected, f"managed environment is not a contained directory: {resolved}"
    return expected, None


def managed_environment_creation_target_issue(name: str) -> str | None:
    """Refuse provisioning through a redirected envs parent or environment leaf."""
    target = paths.env_dir(name)
    root_issue = managed_provisioning_root_issue(create=True)
    if root_issue is not None:
        return root_issue
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        parent = target.parent.resolve(strict=True)
        root = paths.root().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return f"managed environment parent cannot be resolved: {exc}"
    if target.parent.is_symlink() or not parent.is_relative_to(root):
        return f"managed environment parent is redirected outside provisioning root: {parent}"
    if target.is_symlink():
        return f"managed environment path is a symlink: {target}"
    if target.exists() and not target.is_dir():
        return f"managed environment path is not a directory: {target}"
    return None


def managed_url_artifact_path(
    package: Package, value: object,
) -> tuple[Path | None, str | None]:
    """Bind a single-file receipt to its manifest-owned path and provisioning root."""
    expected = paths.models_dir() / str(package.source["filename"])
    candidate = Path(str(value)) if value else None
    if candidate is None:
        return None, "artifact path is absent"
    root_issue = managed_provisioning_root_issue()
    if root_issue is not None:
        return candidate, root_issue
    if Path(os.path.abspath(candidate)) != Path(os.path.abspath(expected)):
        return candidate, f"artifact path {candidate} is not managed path {expected}"
    if expected.parent.is_symlink():
        return candidate, f"managed artifact parent is a symlink: {expected.parent}"
    if candidate.is_symlink():
        return candidate, f"managed artifact path is a symlink: {candidate}"
    try:
        parent = expected.parent.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        root = paths.root().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return candidate, f"managed artifact path cannot be resolved: {exc}"
    if (
        not expected.parent.is_dir()
        or not parent.is_relative_to(root)
        or not candidate.is_file()
        or not resolved.is_relative_to(root)
    ):
        return candidate, f"managed artifact is not a contained regular file: {resolved}"
    return candidate, None


def checkout_file_matches(
    checkout: Path, name: str, digest: str, hasher=sha256_file,
) -> bool:
    """Require a real in-checkout regular file before comparing its pinned digest."""
    target = checkout / name
    try:
        checkout_root = checkout.resolve(strict=True)
        resolved = target.resolve(strict=True)
        return (
            target.is_file()
            and not target.is_symlink()
            and resolved.is_relative_to(checkout_root)
            and hasher(target) == digest
        )
    except (OSError, RuntimeError):
        return False


def _checkout_integrity_issues(
    package: Package,
    materialized: dict,
    toolchain: Toolchain,
) -> list[str]:
    """Verify a native backend's live checkout instead of trusting its pull receipt."""
    specification = package.checkout
    if specification is None:
        return []

    issues: list[str] = []
    checkout, location_issue = managed_checkout_path(
        package, materialized.get("checkout")
    )
    if location_issue is not None:
        return [location_issue]
    assert checkout is not None
    if not checkout.is_dir():
        return [f"source checkout is not a directory: {checkout}"]

    expected_commit = specification.get("resolved_commit", specification["commit"])
    accepted_receipts = (specification["commit"], expected_commit)
    recorded_commit = materialized.get("checkout_commit")
    if recorded_commit not in accepted_receipts:
        issues.append(
            f"recorded checkout commit is {recorded_commit!r}, expected one of "
            f"{list(accepted_receipts)!r}"
        )
    try:
        state = toolchain.inspect_checkout(checkout)
    except ValueError as exc:
        return [str(exc)]
    if state.head != expected_commit:
        issues.append(
            f"checkout HEAD is {state.head!r}, expected exact commit {expected_commit!r}"
        )

    try:
        expected_patches, expected_names, expected_digests = (
            checkout_patch_expectation(package)
        )
    except (OSError, ValueError) as exc:
        return [f"shipped checkout patch cannot be read: {exc}"]
    applied = materialized.get("patches_applied", [])
    if not isinstance(applied, list) or applied != list(expected_patches):
        issues.append(
            f"recorded patches are {applied!r}, expected exactly {list(expected_patches)!r}"
        )

    expected_modified = set(expected_names)
    recorded = materialized.get("patched_file_digests", {})
    if not isinstance(recorded, dict) or recorded != expected_digests:
        issues.append(
            f"recorded patched-file digests are {recorded!r}, expected manifest values "
            f"{expected_digests!r}"
        )
    changed: list[str] = []
    for name, digest in expected_digests.items():
        if not checkout_file_matches(checkout, name, digest, toolchain.file_digest):
            changed.append(name)
    if changed:
        issues.append(f"live patched-file hashes changed for {sorted(changed)!r}")

    # Exact Git names reject added changes while manifest-owned hashes bind the permitted files'
    # contents without depending on Git's configurable/version-dependent diff presentation.
    if set(state.modified) != expected_modified:
        issues.append(
            f"tracked checkout changes are {list(state.modified)!r}, expected exactly "
            f"{sorted(expected_modified)!r}"
        )
    if state.untracked:
        issues.append(
            f"checkout has ordinary or ignored untracked files: {list(state.untracked)!r}"
        )
    return issues


def built_product_candidates(checkout: Path, product: str) -> list[Path]:
    """Contained non-symlink executable products from the pinned release build."""
    try:
        checkout_root = checkout.resolve(strict=True)
    except (OSError, RuntimeError):
        return []
    found: set[Path] = set()
    for path in checkout.glob(f".build/**/release/{product}"):
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if (
            path.is_file()
            and not path.is_symlink()
            and resolved.is_relative_to(checkout_root)
            and os.access(path, os.X_OK)
        ):
            found.add(resolved)
    return sorted(found)


def validated_built_product(
    checkout: Path,
    product: str,
    materialized: dict,
) -> tuple[Path | None, str | None]:
    """Bind the launchable product to the exact bytes recorded after `pull` built it."""
    recorded_path = materialized.get("product_path")
    recorded_digest = materialized.get("product_sha256")
    if not isinstance(recorded_path, str) or not recorded_path:
        return None, "built product receipt has no non-empty product_path"
    if (
        not isinstance(recorded_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", recorded_digest) is None
    ):
        return None, "built product receipt has no valid product_sha256"
    candidates = built_product_candidates(checkout, product)
    if len(candidates) != 1:
        return None, (
            f"expected one executable built product {product!r}, found {len(candidates)}"
        )
    executable = candidates[0]
    try:
        relative = executable.relative_to(checkout.resolve(strict=True)).as_posix()
    except (OSError, RuntimeError, ValueError) as exc:
        return None, f"built product path cannot be bound to its checkout: {exc}"
    if relative != recorded_path:
        return None, (
            f"built product path is {relative!r}, expected receipt path {recorded_path!r}"
        )
    try:
        actual_digest = sha256_file(executable)
    except OSError as exc:
        return None, f"built product digest could not be read: {exc}"
    if actual_digest != recorded_digest:
        return None, (
            f"built product sha256 is {actual_digest!r}, expected receipt digest "
            f"{recorded_digest!r}"
        )
    return executable, None


def _environment_has_built_runtime(name: str, document: dict) -> bool:
    """Whether an environment can run without the tool that provisioned its product."""
    if name != "swift":
        return False
    entry = document.get("packages", {}).get("fluidaudio", {})
    if entry.get("state") != "ready":
        return False
    materialized = entry.get("materialized", {})
    checkout_value = materialized.get("path")
    if not checkout_value:
        return False
    package = packages()["fluidaudio"]
    checkout, issue = managed_checkout_path(package, checkout_value)
    if issue is not None or checkout is None:
        return False
    executable, product_issue = validated_built_product(
        checkout, str(package.source["product"]), materialized
    )
    return product_issue is None and executable is not None


def _environment_built_runtime_runs(
    name: str,
    document: dict,
    toolchain: Toolchain,
) -> bool:
    """Live runtime exemption from a missing provisioning tool."""
    if not _environment_has_built_runtime(name, document):
        return False
    entry = document["packages"]["fluidaudio"]
    package = packages()["fluidaudio"]
    checkout, issue = managed_checkout_path(package, entry["materialized"].get("path"))
    if issue is not None or checkout is None:
        return False
    try:
        state = toolchain.inspect_checkout(checkout)
    except ValueError:
        return False
    try:
        expected_patches, expected_names, expected_digests = (
            checkout_patch_expectation(package)
        )
    except (OSError, ValueError):
        return False
    materialized = entry["materialized"]
    if (
        state.head != package.source["commit"]
        or set(state.modified) != set(expected_names)
        or materialized.get("patches_applied", []) != list(expected_patches)
        or materialized.get("patched_file_digests", {}) != expected_digests
        or any(
            not checkout_file_matches(checkout, name, digest, toolchain.file_digest)
            for name, digest in expected_digests.items()
        )
    ):
        return False
    product = str(package.source["product"])
    executable, product_issue = validated_built_product(
        checkout, product, materialized
    )
    if product_issue is not None or executable is None:
        return False
    try:
        return toolchain.built_product_runs(executable)
    except OSError:
        return False


def _toolchain_missing(package: Package, tool: str) -> ProvisioningError:
    """The exit-3 refusal for a package whose external toolchain is absent."""
    return ProvisioningError(
        "toolchain_missing", f"{package.id} needs {tool}, which is not on PATH",
        missing_tool=tool, package=package.id, requires_tool=[tool],
    )


def _source_revision_report(package: Package) -> dict:
    """What a Hub package's `verify` entry can honestly claim, which is not a digest.

    Nothing here hashes a snapshot. The manifest pins a revision and carries no `sha256` for a Hub
    source, so there is nothing to hash *against*; a `digest` key would name a check no code
    performs. The revision is what is pinned, and the existence check above it is the rest of what
    was verified. So the two claims are told apart by which key is present — `digest` where
    contents were hashed against a manifest pin, `revision`/`revisions` where a revision is pinned
    and the snapshot is present — rather than by a `digest_verified: false` confession.
    """
    revisions = _source_revisions(package)
    if len(revisions) == 1:
        return {"revision": revisions[0]}
    if revisions:
        return {"revisions": revisions}
    return {}


def _source_revisions(package: Package) -> list[str]:
    """Every Hub revision a package pins, whether it names one or four."""
    source = package.source
    if source["type"] == "huggingface":
        return [source["revision"]]
    if source["type"] == "huggingface_multi":
        return [repo["revision"] for repo in source["repos"]]
    return []


def _managed_package_locations(package: Package | None) -> list[Path]:
    """Local deletion targets derived only from the installed manifest."""
    if package is None:
        return []
    found: list[Path] = []
    if package.checkout is not None or package.source["type"] == "git+build":
        found.append(paths.checkout_dir(package.environment, package.id))
    if package.source["type"] == "url":
        found.append(paths.models_dir() / str(package.source["filename"]))
    return found


def _owned_tree_bytes(target: Path) -> int:
    """Bytes below a managed target without following any symlink."""
    try:
        if target.is_symlink():
            return target.lstat().st_size
        if target.is_file():
            return target.stat().st_size
        if not target.is_dir():
            return 0
    except OSError:
        return 0
    total = 0
    for directory, directories, filenames in os.walk(target, followlinks=False):
        base = Path(directory)
        for name in [*directories, *filenames]:
            item = base / name
            try:
                if item.is_symlink():
                    total += item.lstat().st_size
                elif item.is_file():
                    total += item.stat().st_size
            except OSError:
                continue
    return total


def _owned_tree_bytes_at(parent_descriptor: int, name: str) -> int:
    """Measure one descriptor-relative tree without following any symlink."""

    try:
        found = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return 0
    if stat.S_ISLNK(found.st_mode) or stat.S_ISREG(found.st_mode):
        return found.st_size
    if not stat.S_ISDIR(found.st_mode):
        return 0
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    directory_descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    try:
        total = 0
        with os.scandir(directory_descriptor) as entries:
            for entry in entries:
                try:
                    total += _owned_tree_bytes_at(
                        directory_descriptor, entry.name
                    )
                except FileNotFoundError:
                    continue
        return total
    finally:
        os.close(directory_descriptor)


def _delete_at(parent_descriptor: int, name: str) -> None:
    """Delete one descriptor-relative leaf without following it outside its parent."""

    try:
        found = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(found.st_mode):
        if not shutil.rmtree.avoids_symlink_attacks:
            raise OSError("platform recursive deletion is not symlink-attack resistant")
        shutil.rmtree(name, dir_fd=parent_descriptor)
    else:
        os.unlink(name, dir_fd=parent_descriptor)


def _delete_managed(target: Path) -> int:
    """Delete through a no-follow parent descriptor, then confirm descriptor-relative absence."""
    root = Path(os.path.abspath(paths.root()))
    absolute = Path(os.path.abspath(target))
    if absolute == root or not absolute.is_relative_to(root):
        raise ProvisioningError(
            "delete_refused", f"managed deletion target is outside the provisioning root: {target}",
            target=str(target), fix="Inspect the provisioning registry and managed cache root",
        )
    opened = False
    try:
        with bound_directory(
            absolute.parent,
            root=root,
            create=False,
            # Once opened, the descriptor owns the safe operation. If an attacker renames
            # the parent afterward, deleting from that original directory is still contained;
            # following its replacement pathname would not be.
            verify_on_exit=False,
        ) as parent_descriptor:
            opened = True
            before = _owned_tree_bytes_at(parent_descriptor, absolute.name)
            _delete_at(parent_descriptor, absolute.name)
            try:
                os.stat(
                    absolute.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return before
            raise OSError("managed target still exists after deletion")
    except FileNotFoundError:
        return 0
    except OSError as exc:
        if not opened:
            raise ProvisioningError(
                "delete_refused",
                f"managed deletion target cannot be opened without following links: {target}: "
                f"{exc}",
                target=str(target),
                fix=f"Remove or replace redirected parent paths for {target} and retry",
            ) from exc
        raise ProvisioningError(
            "delete_failed", f"could not delete managed target {target}: {exc}",
            target=str(target), fix=f"Restore access to {target} and retry",
        ) from exc


def _teardown_revisions(
    package: Package | None, materialized: dict,
) -> tuple[list[str], list[str]]:
    """Bound mutable ownership receipts to revisions shipped for one known package."""
    allowed = set(_source_revisions(package)) if package is not None else set()

    def strings(value: object) -> set[str]:
        return {item for item in value if isinstance(item, str)} \
            if isinstance(value, list) else set()

    claimed = strings(materialized.get("hub_revisions"))
    pre_existing = strings(materialized.get("hub_revisions_pre_existing"))
    deletable = (claimed & allowed) - pre_existing
    retained = pre_existing | (claimed - allowed)
    return sorted(deletable), sorted(retained)


def _users_by_environment(document: dict) -> dict[str, set[str]]:
    users: dict[str, set[str]] = {}
    catalog = packages()
    for identifier in document["packages"]:
        package = catalog.get(identifier)
        if package is not None:
            users.setdefault(package.environment, set()).add(identifier)
    return users


def _selection_bytes(selection: list[Package], document: dict) -> tuple[int, list[str]]:
    known = 0
    unsized: list[str] = []
    for package in selection:
        materialized = document["packages"].get(package.id, {}).get("materialized", {})
        materialized = materialized if isinstance(materialized, dict) else {}
        if package.source["type"] in {"huggingface", "huggingface_multi"}:
            owned, _retained = _teardown_revisions(package, materialized)
            if package.source["type"] == "huggingface":
                if package.source["revision"] in owned and package.bytes is not None:
                    known += package.bytes
            else:
                known += sum(
                    int(repository.get("bytes") or 0)
                    for repository in package.source["repos"]
                    if repository["revision"] in owned
                )
            continue
        recorded = materialized.get("bytes")
        size = recorded if recorded is not None else package.bytes
        if size is None:
            unsized.append(package.id)
        else:
            known += size
    return known, sorted(unsized)
