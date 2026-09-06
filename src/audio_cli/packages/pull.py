"""Environment creation and package materialization state machine."""

from __future__ import annotations

from .. import paths
from ..environments import Package, environments
from . import catalog, environment_verification, materialization, registry
from .fetcher import Fetcher
from .integrity import _now, sha256_file
from .locations import (
    managed_environment_creation_target_issue,
    managed_provisioning_root_issue,
)
from .models import ProvisioningError
from .reports import is_ready
from .requirements import _environment_drift, _locked_versions, managed_checkout_requirements
from .teardown import (
    _selection_bytes,
    _source_revisions,
    _toolchain_missing,
)
from .toolchain import Toolchain


def ensure_environment(
    toolchain: Toolchain,
    name: str,
    document: dict,
    *,
    repair_fix: str = "audio packages verify --repair",
    repair: bool = False,
    replacing_package: str | None = None,
) -> bool:
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
            "state": "ready",
            "path": str(target),
            "python": None,
            "lock_sha256": None,
            "created_utc": _now(),
        }
        registry.save_registry(document)
        return True

    entry = document["environments"].get(name, {})
    lock_digest = sha256_file(environment.lock)
    if entry.get("state") == "ready" and entry.get("lock_sha256") == lock_digest:
        required = managed_checkout_requirements(document, name)
        drift = _environment_drift(
            _locked_versions(environment),
            toolchain.frozen_packages(paths.env_python(name)),
            required,
        )
        if not drift:
            return False
        if not repair:
            _require_environment_installation(
                toolchain,
                name,
                required,
                fix=f"run audio packages verify --repair, then {repair_fix}",
            )
            return False
        # Explicit pull repair replaces its selected checkout after environment sync.
        # Downgrade it first so syncing never executes the very source being repaired.
        # Other ready native installs retain their existing integrity-before-build gate.
        previous = document["packages"].get(replacing_package)
        if previous is not None:
            previous["state"] = "pulling"
            registry.save_registry(document)
        _states, _probes, _invalid, failed = environment_verification.verify_environments(
            toolchain, document, catalog.packages(), repair=True, environment_names={name}
        )
        if failed:
            failure = failed[0]
            raise ProvisioningError(
                failure["code"],
                failure["detail"],
                environment=name,
                fix=f"run audio packages verify --repair, then {repair_fix}",
            )
        # Dropping the selected package's missing direct install from the required
        # set can resolve the discrepancy without a sync. Only an actual recreation
        # replaces this environment entry; do not report an untouched one as created.
        return document["environments"].get(name) is not entry

    # Intent first: a crash between here and the flip leaves a `creating` entry, which
    # reads as absent and as reclaimable rather than as a working environment.
    document["environments"][name] = {
        "state": "creating",
        "path": str(target),
        "python": environment.python,
        "lock_sha256": lock_digest,
        "created_utc": _now(),
    }
    registry.save_registry(document)

    toolchain.create_environment(environment, target)
    _require_environment_installation(toolchain, name, {}, fix=repair_fix)
    document["environments"][name]["state"] = "ready"
    registry.save_registry(document)
    return True


def _require_environment_installation(
    toolchain: Toolchain, name: str, required_checkouts: dict, *, fix: str
) -> None:
    environment = environments()[name]
    if not environment.has_interpreter:
        return
    drift = _environment_drift(
        _locked_versions(environment),
        toolchain.frozen_packages(paths.env_python(name)),
        required_checkouts,
    )
    if drift:
        raise ProvisioningError(
            "environment_drifted",
            f"{len(drift)} package(s) differ from {environment.lock.name} or required checkout installs",
            environment=name,
            examples={
                n: {"locked": drift[n][0], "installed": drift[n][1]} for n in sorted(drift)[:5]
            },
            fix=fix,
        )


# -- packages ----------------------------------------------------------------------


def pull(
    toolchain: Toolchain,
    fetcher: Fetcher,
    selection: list[Package],
    *,
    repair: bool = False,
    stack: str | None = None,
) -> dict:
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
    document = registry.load_registry()
    # Read the cache once, before anything downloads. Everything after this point works
    # from that snapshot, so a revision fetched by this pull is never mistaken for one that
    # was already there.
    cached = fetcher.cached_revisions()
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

        missing_tool = next(
            (tool for tool in package.requires_tool if toolchain.which(tool) is None), None
        )
        if missing_tool is not None:
            if stack is None:
                raise _toolchain_missing(package, missing_tool)
            blocked.append((package, missing_tool))
            continue

        repair_fix = f"audio packages pull --repair {package.id}"
        if ensure_environment(
            toolchain,
            package.environment,
            document,
            repair_fix=repair_fix,
            repair=repair,
            replacing_package=package.id,
        ):
            created.append(package.environment)

        previous = document["packages"].get(package.id, {})
        pre_existing = _pre_existing_revisions(package, previous, cached)
        entry = {
            "state": "pulling",
            "environment": package.environment,
            "kind": package.kind,
            "source": package.source,
            "license_declared": package.license_declared,
            "license_reviewed": package.license_reviewed,
            "pulled_utc": _now(),
            # Decided before any bytes move, and carried across a retry: see
            # _pre_existing_revisions for why re-deciding would be wrong.
            "hub_revisions_pre_existing": sorted(pre_existing),
        }
        document["packages"][package.id] = entry
        registry.save_registry(document)

        materialized = materialization.materialize(
            toolchain, fetcher, package, pre_existing, repair=repair
        )
        required = managed_checkout_requirements(document, package.environment)
        if package.checkout is not None:
            required[package.checkout["distribution"]] = paths.checkout_dir(
                package.environment, package.id
            ).absolute()
        _require_environment_installation(toolchain, package.environment, required, fix=repair_fix)
        entry["materialized"] = materialized
        entry["state"] = "ready"
        document["packages"][package.id] = entry
        registry.save_registry(document)

        receipt = {
            "package": package.id,
            "environment": package.environment,
            "bytes": materialized.get("bytes"),
        }
        for key in (
            "revision",
            "git_blob_sha1",
            "revisions",
            "digest_verified",
            "built",
            "product_runs",
            "patches_applied",
        ):
            if key in materialized:
                receipt[key] = materialized[key]
        if materialized.get("hub_revisions_pre_existing"):
            receipt["hub_revisions_pre_existing"] = materialized["hub_revisions_pre_existing"]
            count = len(pre_existing)
            total = len(set(_source_revisions(package)))
            receipt["pre_existing_note"] = (
                f"The listed {count} of {total} pinned revisions were recorded as pre-existing "
                f"in the shared Hugging Face cache ({'all' if count == total else 'some'} "
                "revisions). Teardown here will not delete those revisions. This records "
                "ownership, not complete cache reuse: missing files may still be fetched, "
                "and --repair requests a fresh download. Network bytes are not measured."
            )
        pulled.append(receipt)

    if blocked and not pulled and not skipped:
        # Nothing in the stack was provisionable, so there is no partial success to report
        # and the caller needs the exit code rather than a receipt of an empty pull.
        raise _toolchain_missing(*blocked[0])

    if blocked:
        tools = sorted({tool for _, tool in blocked})
        names = [package.id for package, _ in blocked]
        warnings.append(
            {
                "code": "toolchain_missing",
                "blocking": True,
                "packages": names,
                "requires_tool": tools,
                "detail": (
                    f"{', '.join(names)} {'needs' if len(names) == 1 else 'need'} "
                    f"{', '.join(tools)}, which {'is' if len(tools) == 1 else 'are'} not on "
                    f"PATH, so {'it' if len(names) == 1 else 'they'} "
                    f"{'was' if len(names) == 1 else 'were'} not provisioned; the rest of "
                    f"{f'stack {stack}' if stack else 'the selection'} was. Install the "
                    f"toolchain and pull {'it' if len(names) == 1 else 'them'} by name."
                ),
            }
        )

    # A blocked package provisioned nothing, so it carries no license claim and no bytes.
    # This is ownership-scoped artifact size, not transfer accounting or added disk usage.
    blocked_ids = {package.id for package, _ in blocked}
    provisioned = [p for p in selection if p.id not in blocked_ids]
    unreviewed = sorted(p.id for p in provisioned if not p.license_reviewed)
    if unreviewed:
        warnings.append(
            {
                "code": "license_unreviewed",
                "blocking": False,
                "packages": unreviewed,
                "detail": (
                    f"{', '.join(unreviewed)} "
                    f"{'reports' if len(unreviewed) == 1 else 'report'} a license their model "
                    "card declares but nobody has reviewed. A declared license is evidence "
                    "that one exists, not a redistribution clearance."
                ),
            }
        )

    known, unsized = _selection_bytes(
        [p for p in provisioned if p.id not in set(skipped)], document
    )
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
        "pulled_known_bytes_note": (
            "Known artifact sizes for packages materialized by this invocation, including "
            "repairs: manifest-declared sizes for Hub revisions owned by this root, and "
            "recorded local artifact sizes. Excludes pre-existing shared revisions, skipped "
            "packages, and environment bytes. Not measured network bytes or added disk usage."
        ),
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


def _pre_existing_revisions(package: Package, previous: dict, cached: set[str]) -> set[str]:
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
    current = _hub_source_identities(package.source)
    if "hub_revisions_pre_existing" not in previous:
        return {revision for _repository, revision in current if revision in cached}
    historical = _hub_source_identities(previous.get("source"))
    retained = set(previous["hub_revisions_pre_existing"])
    # A retry's ownership decision applies only to the source identity recorded on
    # that attempt. A newly pinned repository/revision needs its own pre-download
    # cache decision; inheriting the old pin's ownership can delete shared weights.
    return {
        revision
        for repository, revision in current
        if revision in (retained if (repository, revision) in historical else cached)
    }


def _hub_source_identities(source: object) -> set[tuple[str, str]]:
    if not isinstance(source, dict):
        return set()
    repositories = (
        source.get("repos", [])
        if source.get("type") == "huggingface_multi"
        else [source]
        if source.get("type") == "huggingface"
        else []
    )
    if not isinstance(repositories, list):
        return set()
    return {
        (repository["repo"], repository["revision"])
        for repository in repositories
        if isinstance(repository, dict)
        and isinstance(repository.get("repo"), str)
        and isinstance(repository.get("revision"), str)
    }
