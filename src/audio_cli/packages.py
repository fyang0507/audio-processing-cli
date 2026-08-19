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
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import paths
from .environments import Environment, ManifestError, Package, environments, packages

REGISTRY_SCHEMA_VERSION = 1

HUB_CACHE_NOTE = (
    "weights live in the shared Hugging Face cache, not under this root. Only revisions this "
    "pull actually downloaded were deleted; a revision that was already cached is retained, "
    "because it may belong to another tool, another provisioning root, or an earlier experiment"
)
UNTOUCHED = ["user media", "transcript and subtitle outputs"]


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
        for line in result.stdout.splitlines():
            if "==" in line and not line.startswith("#"):
                name, _, version = line.partition("==")
                frozen[name.strip().lower().replace("_", "-")] = version.strip()
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

    def swift_product_runs(self, checkout: Path) -> bool:
        result = self.run(["swift", "run", "-c", "release", "fluidaudio", "--help"],
                          cwd=checkout, timeout=600)
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

    def hf_snapshot(self, repo: str, revision: str, *, force: bool = False) -> Path:
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
        return Path(snapshot_download(repo, revision=revision, force_download=force))

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
        """The vad.py pattern: digest check and atomic rename, kept identical on purpose."""
        if target.is_file() and sha256_file(target) == sha256:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.{os.getpid()}.part")
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "audio-processing-cli/0.1"})
            with (urllib.request.urlopen(request, timeout=60) as response,
                  partial.open("wb") as output):
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            actual = sha256_file(partial)
            if actual != sha256:
                raise ProvisioningError(
                    "package_integrity_failed",
                    f"{target.name} checksum mismatch: expected {sha256}, got {actual}",
                    expected=sha256, actual=actual,
                )
            os.replace(partial, target)
        except (OSError, urllib.error.URLError) as exc:
            raise ProvisioningError("download_failed",
                                    f"could not download {url}: {exc}") from exc
        finally:
            partial.unlink(missing_ok=True)
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


def load_registry() -> dict:
    target = paths.registry_path()
    if not target.is_file():
        return blank_registry()
    try:
        document = json.loads(target.read_text())
    except json.JSONDecodeError as exc:
        raise ProvisioningError(
            "registry_unreadable", f"{target} is not valid JSON: {exc}",
            fix=f"Move {target} aside and run audio packages pull again",
        ) from exc
    if document.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ProvisioningError(
            "registry_unreadable",
            f"{target} has schema_version {document.get('schema_version')!r}, expected "
            f"{REGISTRY_SCHEMA_VERSION}",
        )
    document.setdefault("environments", {})
    document.setdefault("packages", {})
    return document


def save_registry(document: dict) -> None:
    """Atomic rename, so a registry is never observed half-written."""
    target = paths.registry_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    partial.write_text(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(partial, target)


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
        return [catalog[identifier] for identifier in package_ids]
    if stack is not None:
        chosen = [package for package in catalog.values() if stack in package.stacks]
        if not chosen:
            raise ProvisioningError(
                "stack_unknown", f"no packages are registered for stack {stack!r}", exit_code=2,
                allowed=sorted({name for p in catalog.values() for name in p.stacks}),
            )
        return sorted(chosen, key=lambda package: package.id)
    raise ProvisioningError("nothing_selected", "name packages, or pass --stack", exit_code=2)


# --------------------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------------------


def path_report() -> dict:
    """Where everything is, so a session with no provisioning history can still find it."""
    document = load_registry()
    return {
        "root": str(paths.root()),
        "registry": str(paths.registry_path()),
        # Most of the bytes are not under the root, and a reader who assumes they are concludes
        # that a 17 GiB pull silently did nothing. Say where weights actually land, and say what
        # the models directory is for rather than printing a path that is often absent.
        "weights": {
            "location": "the Hugging Face cache, shared with other tools",
            "note": "per-package `location` below is authoritative; the registry records the "
                    "revisions this root materialized there",
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
                "location": entry.get("materialized", {}).get("path"),
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
            "environment": entry.get("environment"),
            "state": entry.get("state"),
            "bytes": size,
            "license_declared": entry.get("license_declared"),
            "license_reviewed": entry.get("license_reviewed", False),
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

        if not environment.has_interpreter:
            # `swift` has no interpreter and so nothing to sync, but it still holds a build
            # product and a Core ML package. It gets a registry entry anyway: reference
            # counting and purge treat every provisioned environment the same way, and an
            # untracked directory is one nothing can reclaim.
            if document["environments"].get(name, {}).get("state") == "ready":
                return False
            target = paths.env_dir(name)
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

        target = paths.env_dir(name)
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
            return {"path": str(resolved), "bytes": _tree_bytes(resolved),
                    "digest_verified": True}

        if kind == "huggingface":
            revision = package.source["revision"]
            snapshot = self.fetcher.hf_snapshot(package.source["repo"], revision, force=repair)
            result = {
                "path": str(snapshot), "bytes": _tree_bytes(snapshot),
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
                snapshot = self.fetcher.hf_snapshot(repo["repo"], repo["revision"],
                                                    force=repair)
                snapshots[repo["repo"]] = str(snapshot)
                total += _tree_bytes(snapshot)
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
            if repair:
                # Rebuilding in place would trust the checkout whose state is what is in doubt:
                # `clone` skips an existing directory, so a dirty or half-built tree survives.
                _delete(checkout)
            checkout.parent.mkdir(parents=True, exist_ok=True)
            self.toolchain.clone(package.source["repo"], package.source["commit"], checkout)
            self.toolchain.swift_build(checkout)
            return {"path": str(checkout), "bytes": _tree_bytes(checkout / ".build"),
                    "revision": package.source["commit"], "built": True,
                    "product_runs": self.toolchain.swift_product_runs(checkout)}

        raise ManifestError(f"{package.id}: unsupported source type {kind!r}")

    def _checkout_and_install(self, package: Package, *, repair: bool = False) -> dict:
        """Pinned source checkout, patch, and a --no-deps install into the environment."""
        if package.checkout is None:
            return {}
        checkout = paths.checkout_dir(package.environment, package.id)
        if repair:
            # Same reason as the Swift build: a repair of `patch_not_applied` must not re-patch a
            # tree that may also be at the wrong commit. Re-clone, then patch.
            _delete(checkout)
        checkout.parent.mkdir(parents=True, exist_ok=True)
        self.toolchain.clone(package.checkout["repo"], package.checkout["commit"], checkout)

        applied: list[str] = []
        digests: dict[str, str] = {}
        patch_name = package.checkout.get("patch")
        if patch_name:
            from .environments import HERE

            patch = HERE / patch_name
            if not patch.is_file():
                raise ProvisioningError("patch_missing", f"{patch} is not in the installed wheel",
                                        patch=patch_name, package=package.id)
            self.toolchain.apply_patch(checkout, patch)
            applied.append(Path(patch_name).name)
            for touched in _patched_files(patch, checkout):
                if touched.is_file():
                    digests[str(touched.relative_to(checkout))] = sha256_file(touched)

        self.toolchain.install_checkout(paths.env_python(package.environment), checkout)
        return {"checkout": str(checkout), "checkout_commit": package.checkout["commit"],
                "patches_applied": applied, "patched_file_digests": digests}

    # -- verify ------------------------------------------------------------------------

    def verify(self, *, repair: bool = False) -> dict:
        document = load_registry()
        catalog = packages()
        verified: list[dict] = []
        failed: list[dict] = []
        environment_states: dict[str, str] = {}

        for name, environment in environments().items():
            if not environment.provisioned:
                continue
            entry = document["environments"].get(name)
            if entry is None or entry.get("state") != "ready":
                environment_states[name] = "absent"
                continue
            # A missing toolchain is a property of the environment, not of what is inside it.
            # This tool launches the Swift product through `swift run`, so with no toolchain
            # nothing in that environment can execute whatever the registry holds — and the
            # registry can legitimately hold something, because a stack pull provisions the
            # packages that need no toolchain and reports the blocked one. `ok` there would tell
            # a caller diarization is available on a machine that cannot run it. Not a `failed`
            # entry: nothing provisioned is broken, the gap is a package `list` already reports
            # as absent, and no `audio` command installs a toolchain for a `fix` to name.
            blocked_by = [tool for tool in environment.requires_tool
                          if self.toolchain.which(tool) is None]
            if blocked_by:
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
            drift = {n: (v, frozen.get(n)) for n, v in expected.items() if frozen.get(n) != v}
            if drift and repair:
                self.toolchain.create_environment(environment, paths.env_dir(name))
                frozen = self.toolchain.frozen_packages(paths.env_python(name))
                drift = {n: (v, frozen.get(n)) for n, v in expected.items()
                         if frozen.get(n) != v}
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
            record: dict = {"package": identifier}
            materialized = entry.get("materialized", {})
            # `digest: "ok"` is reserved for the one kind that has something to hash against.
            if package is not None and package.source["type"] == "url":
                location = Path(materialized.get("path", ""))
                if location.is_file() and sha256_file(location) == package.source["sha256"]:
                    record["digest"] = "ok"
                else:
                    failed.append({
                        "package": identifier, "code": "package_integrity_failed",
                        "detail": f"{location} is missing or its digest changed",
                        "fix": f"audio packages pull --repair {identifier}",
                    })
                    continue
            elif "product_runs" in materialized:
                record["product_runs"] = materialized["product_runs"]
                record["patches_applied"] = materialized.get("patches_applied", [])
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
                record.update(_pinned_revisions(materialized))

            digests = materialized.get("patched_file_digests") or {}
            if digests:
                checkout = Path(materialized["checkout"])
                reverted = [name for name, expected in digests.items()
                            if not (checkout / name).is_file()
                            or sha256_file(checkout / name) != expected]
                if reverted:
                    failed.append({
                        "package": identifier, "code": "patch_not_applied",
                        "detail": f"patched file(s) no longer match: {', '.join(reverted)}",
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

        for identifier in targets:
            materialized = document["packages"][identifier].get("materialized", {})
            hub_revisions.extend(materialized.get("hub_revisions", []))
            retained.extend(materialized.get("hub_revisions_pre_existing", []))
            # Measured before deleting, not read off the registry: what a teardown reports as
            # reclaimed has to be what the filesystem actually gave back.
            for location in _locations(materialized):
                local_freed += _tree_bytes(location) if location.exists() else 0
                _delete(location)
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
                "already in the Hugging Face cache before this root pulled them, so they are "
                "not this root's to delete"
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
        for name in sorted(document["environments"]):
            if users.get(name):
                kept.append(name)
                continue
            target = paths.env_dir(name)
            freed += _tree_bytes(target) if target.exists() else 0
            _delete(target)
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
            for identifier in package_ids:
                materialized = document["packages"][identifier].get("materialized", {})
                deletable.extend(materialized.get("hub_revisions", []))
                keeping.extend(materialized.get("hub_revisions_pre_existing", []))
            return {
                "would_remove": {"packages": package_ids, "environments": environment_names,
                                 "root": str(paths.root()),
                                 "hub_revisions": sorted(set(deletable))},
                "would_keep": {"hub_revisions": sorted(set(keeping))},
                "hub_cache_note": HUB_CACHE_NOTE,
                "reclaimable_known_bytes": known,
                "reclaimable_note": (
                    "projected from the registry, and it counts weights this root downloaded "
                    "plus what is under the root; a retained revision is not included"
                ),
                "unsized_packages": unsized,
                "untouched": UNTOUCHED,
            }

        hub_revisions: list[str] = []
        retained: list[str] = []
        local_freed = 0
        # `purge` cannot take a name that is not in the registry — it reads the list *from* the
        # registry — so it never had `remove`'s validation defect. It shared the narrower half:
        # every local file was deleted and the registry was cleared in one write afterwards, so
        # anything that raised in between left every package reading as `ready` with nothing
        # behind it. Same rule as `remove`, then: an entry goes as soon as its own bytes do.
        for identifier in package_ids:
            materialized = document["packages"][identifier].get("materialized", {})
            hub_revisions.extend(materialized.get("hub_revisions", []))
            retained.extend(materialized.get("hub_revisions_pre_existing", []))
            for location in _locations(materialized):
                local_freed += _tree_bytes(location) if location.exists() else 0
                _delete(location)
            document["packages"].pop(identifier)
            save_registry(document)
        for name in environment_names:
            local_freed += _tree_bytes(paths.env_dir(name)) if paths.env_dir(name).exists() else 0
            _delete(paths.env_dir(name))
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
                    tool for tool in environment.requires_tool if not tools[tool]["present"]
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
            "An absent swift blocks only the packages that need it; it is reported rather than "
            "fatal."
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
        versions[name.strip().lower().replace("_", "-")] = rest.split()[0].strip(" \\")
    return versions


def _patched_files(patch: Path, checkout: Path) -> list[Path]:
    """Files a unified diff touches, so verify can detect a reverted patch."""
    touched = []
    for line in patch.read_text(errors="replace").splitlines():
        if line.startswith("+++ ") and not line.endswith("/dev/null"):
            target = line[4:].strip()
            if target.startswith("b/"):
                target = target[2:]
            touched.append(checkout / target)
    return touched


def _toolchain_missing(package: Package, tool: str) -> ProvisioningError:
    """The exit-3 refusal for a package whose external toolchain is absent."""
    return ProvisioningError(
        "toolchain_missing", f"{package.id} needs {tool}, which is not on PATH",
        missing_tool=tool, package=package.id, requires_tool=[tool],
    )


def _pinned_revisions(materialized: dict) -> dict:
    """What a Hub package's `verify` entry can honestly claim, which is not a digest.

    Nothing here hashes a snapshot. The manifest pins a revision and carries no `sha256` for a Hub
    source, so there is nothing to hash *against*; a `digest` key would name a check no code
    performs. The revision is what is pinned, and the existence check above it is the rest of what
    was verified. So the two claims are told apart by which key is present — `digest` where
    contents were hashed against a manifest pin, `revision`/`revisions` where a revision is pinned
    and the snapshot is present — rather than by a `digest_verified: false` confession.
    """
    if "revision" in materialized:
        return {"revision": materialized["revision"]}
    if "revisions" in materialized:
        return {"revisions": list(materialized["revisions"])}
    return {}


def _source_revisions(package: Package) -> list[str]:
    """Every Hub revision a package pins, whether it names one or four."""
    source = package.source
    if source["type"] == "huggingface":
        return [source["revision"]]
    if source["type"] == "huggingface_multi":
        return [repo["revision"] for repo in source["repos"]]
    return []


def _locations(materialized: dict) -> list[Path]:
    """Filesystem locations this tool created. Hub snapshots are excluded on purpose.

    Deleting a Hub snapshot directory would reach into a cache other tools may share; the
    revisions are reported instead, which is what `hub_cache_note` says.
    """
    found = []
    for key in ("checkout",):
        if materialized.get(key):
            found.append(Path(materialized[key]))
    path = materialized.get("path")
    if path and _inside(Path(path), paths.models_dir()):
        found.append(Path(path))
    return found


def _inside(candidate: Path, directory: Path) -> bool:
    """Real path containment, because the substring test this replaces was not one.

    `str(models_dir) in path` also matched a *sibling* whose name starts with the models
    directory's — point `HF_HOME` at `<root>/models_hub` and every snapshot path under it tests
    positive, so teardown would delete directories inside the shared Hub cache that
    `_locations` exists to keep its hands off.
    """
    return candidate != directory and candidate.is_relative_to(directory)


def _delete(target: Path) -> None:
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target, ignore_errors=True)
    elif target.exists() or target.is_symlink():
        target.unlink(missing_ok=True)


def _users_by_environment(document: dict) -> dict[str, set[str]]:
    users: dict[str, set[str]] = {}
    for identifier, entry in document["packages"].items():
        users.setdefault(entry.get("environment", ""), set()).add(identifier)
    return users


def _selection_bytes(selection: list[Package], document: dict) -> tuple[int, list[str]]:
    known = 0
    unsized: list[str] = []
    for package in selection:
        recorded = document["packages"].get(package.id, {}).get("materialized", {}).get("bytes")
        size = recorded if recorded is not None else package.bytes
        if size is None:
            unsized.append(package.id)
        else:
            known += size
    return known, sorted(unsized)
