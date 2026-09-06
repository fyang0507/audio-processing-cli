"""Validation of immutable single-file source declarations; no installed-state access."""

from __future__ import annotations

import re


def git_blob_source_problems(
    source: dict, byte_count: int | None, *, auto_fetch: bool
) -> list[str]:
    """Require a GitHub raw URL derived exactly from a pinned Git tree entry."""
    problems = []
    for key in ("revision", "git_blob_sha1"):
        if not isinstance(source.get(key), str) or not re.fullmatch("[0-9a-f]{40}", source[key]):
            problems.append(f"git-blob {key} must be a full lowercase Git object identity")
    repo = source.get("repo")
    if not isinstance(repo, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo
    ):
        problems.append("git-blob repo must name its GitHub owner/repository")
    path = source.get("path")
    if (
        not isinstance(path, str)
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or not re.fullmatch(r"[A-Za-z0-9_./-]+", path)
    ):
        problems.append("git-blob path must be a safe repository-relative path")
    expected_url = f"https://raw.githubusercontent.com/{repo}/{source.get('revision')}/{path}"
    if source.get("url") != expected_url:
        problems.append("git-blob url must be the exact HTTPS repo/revision/path URL")
    filename = source.get("filename")
    if (
        not isinstance(filename, str)
        or filename in {".", ".."}
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", filename)
    ):
        problems.append("git-blob filename must be one safe managed filename")
    if type(byte_count) is not int or byte_count <= 0:
        problems.append("git-blob bytes must be a positive actual artifact size")
    if "sha256" in source:
        problems.append("git-blob uses Git SHA-1 content identity, not a SHA-256 declaration")
    if auto_fetch:
        problems.append("git-blob artifacts must be explicitly provisioned, never auto-fetched")
    return problems


__all__ = ["git_blob_source_problems"]
