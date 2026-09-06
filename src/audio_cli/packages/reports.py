"""Read-only selection, path, and inventory reports for packages."""

from __future__ import annotations

from .. import paths
from ..environments import Package, environments
from . import catalog, registry
from .models import ProvisioningError


def is_ready(document: dict, package_id: str) -> bool:
    return document.get("packages", {}).get(package_id, {}).get("state") == "ready"


def missing_packages(selection: list[Package]) -> list[Package]:
    """What `run` reports at exit 3. Anything not `ready` is absent, including a crashed pull."""
    document = registry.load_registry()
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
    package_catalog = catalog.packages()
    if package_ids and stack is not None:
        raise ProvisioningError(
            "stack_conflicts_with_named_packages",
            f"--stack {stack} was passed alongside named packages; a stack selects every package "
            "it can use and named ids select exactly those, so one of the two has to go",
            exit_code=2,
            stack=stack,
            packages=list(package_ids),
            fix=f"audio packages pull {' '.join(package_ids)}",
        )
    if package_ids:
        unknown = sorted(set(package_ids) - set(package_catalog))
        if unknown:
            raise ProvisioningError(
                "package_unknown",
                f"no such package: {', '.join(unknown)}",
                exit_code=2,
                allowed=sorted(package_catalog),
            )
        # A repeated positional id is still one package selection.  Besides duplicating the
        # receipt, preserving duplicates is destructive under ``--repair``: the same weights are
        # force-downloaded or the same checkout is rebuilt twice.  ``dict`` retains the caller's
        # first-occurrence order while making the selection a stable set.
        return [package_catalog[identifier] for identifier in dict.fromkeys(package_ids)]
    if stack is not None:
        chosen = [package for package in package_catalog.values() if stack in package.stacks]
        if not chosen:
            raise ProvisioningError(
                "stack_unknown",
                f"no packages are registered for stack {stack!r}",
                exit_code=2,
                allowed=sorted(
                    {name for package in package_catalog.values() for name in package.stacks}
                ),
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
    document = registry.load_registry()
    package_catalog = catalog.packages()
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
            "holds": "pinned single-file artifacts only: silero-vad and rnnoise-voice",
        },
        "environments": {
            name: {
                "path": str(paths.env_dir(name)),
                "python": str(paths.env_python(name)) if environment.has_interpreter else None,
                "state": document["environments"].get(name, {}).get("state", "absent"),
            }
            for name, environment in environments().items()
            if environment.provisioned
        },
        "packages": {
            identifier: {
                "state": entry.get("state", "absent"),
                **(
                    {
                        "location": str(
                            paths.models_dir() / str(package_catalog[identifier].source["filename"])
                        )
                    }
                    if identifier in package_catalog
                    and package_catalog[identifier].source["type"] in {"url", "git-blob"}
                    else _path_location_fields(entry)
                ),
            }
            for identifier, entry in sorted(document["packages"].items())
        },
    }


def list_report() -> dict:
    document = registry.load_registry()
    package_catalog = catalog.packages()
    listed = []
    total_known = 0
    unsized: list[str] = []
    for identifier, entry in sorted(document["packages"].items()):
        package = package_catalog.get(identifier)
        size = entry.get("materialized", {}).get("bytes")
        if size is None and package is not None:
            size = package.bytes
        if size is None:
            unsized.append(identifier)
        else:
            total_known += size
        listed.append(
            {
                "package": identifier,
                "environment": package.environment if package else entry.get("environment"),
                "state": entry.get("state"),
                "bytes": size,
                "license_declared": (
                    package.license_declared if package else entry.get("license_declared")
                ),
                "license_reviewed": (
                    package.license_reviewed if package else entry.get("license_reviewed", False)
                ),
                "used_by_stacks": list(package.stacks) if package else [],
            }
        )
    return {
        "root": str(paths.root()),
        "packages": listed,
        "environments": {
            name: document["environments"].get(name, {}).get("state", "absent")
            for name, environment in environments().items()
            if environment.provisioned
        },
        "total_known_bytes": total_known,
        "unsized_packages": unsized,
    }
