"""Live verification and repair orchestration for provisioned packages."""

from __future__ import annotations

import json

from .. import paths
from ..environments import environments
from . import catalog, environment_verification, package_verification, registry
from .locations import managed_environment_path
from .requirements import (
    _class_of,
    _method_of,
    _module_of,
)
from .toolchain import Toolchain


def _verify_mlx_guard(toolchain: Toolchain, document: dict) -> dict:
    """Check the pinned private decode path without loading a checkpoint."""
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
    result = toolchain.run([str(paths.env_python("mlx")), "-c", probe])
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
    report["mlx_audio_private_api_signature_ok"] = set(signature["required_parameters"]).issubset(
        observed["params"]
    )
    report["mlx_audio_private_api_target"] = target
    return report


def verify(toolchain: Toolchain, *, repair: bool = False) -> dict:
    document = registry.load_registry()
    package_catalog = catalog.packages()
    (
        environment_states,
        built_runtime_probes,
        invalid_environment_roots,
        failed,
    ) = environment_verification.verify_environments(
        toolchain, document, package_catalog, repair=repair
    )
    verified, package_failures = package_verification.verify_packages(
        toolchain,
        document,
        package_catalog,
        built_runtime_probes,
        invalid_environment_roots,
    )
    failed.extend(package_failures)
    report: dict = {
        "verified": verified,
        "environments": environment_states,
        "failed": failed,
    }
    report.update(_verify_mlx_guard(toolchain, document))
    return report
