"""Verify the CLI user skill's layout, portable links, and built source distribution."""

from __future__ import annotations

import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

import pytest

REPO = Path(__file__).resolve().parents[2]
SKILL = REPO / "skills" / "audio-cli"


def test_user_skill_is_not_exposed_as_development_agent_instructions() -> None:
    assert (SKILL / "SKILL.md").is_file()
    assert (SKILL / "agents" / "openai.yaml").is_file()
    assert not SKILL.is_symlink()
    development_skill = REPO / ".agents" / "skills" / "audio-cli"
    assert not development_skill.exists()
    assert not development_skill.is_symlink(), (
        "even a dangling compatibility link exposes the skill"
    )


def test_skill_markdown_links_resolve_inside_the_shipped_bundle() -> None:
    links = [
        (path, target)
        for path in SKILL.rglob("*.md")
        for target in re.findall(r"\[[^\]\n]+\]\(([^)\s]+)\)", path.read_text(encoding="utf-8"))
    ]
    assert links, "the entry point must route to task references"
    for path, target in links:
        resolved = (path.parent / target).resolve()
        assert resolved.is_relative_to(SKILL.resolve()), (path, target)
        assert resolved.is_file(), (path, target)


@pytest.mark.parametrize("document", ["AGENTS.md", "README.md", "docs/README.md"])
def test_current_docs_link_to_the_shipped_skill(document: str) -> None:
    path = REPO / document
    targets = re.findall(
        r"\[[^\]\n]+\]\(([^)\s]*skills/audio-cli/SKILL\.md)\)",
        path.read_text(encoding="utf-8"),
    )
    assert targets, f"{document} must link to the CLI user skill"
    assert all((path.parent / target).resolve() == SKILL / "SKILL.md" for target in targets)


def test_sdist_contains_the_complete_skill_only_at_the_shipped_path(tmp_path: Path) -> None:
    completed = subprocess.run(
        ["uv", "build", "--sdist", "--out-dir", str(tmp_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    archives = list(tmp_path.glob("*.tar.gz"))
    assert len(archives) == 1
    source_files = {
        path.relative_to(REPO).as_posix(): path for path in SKILL.rglob("*") if path.is_file()
    }
    assert "skills/audio-cli/SKILL.md" in source_files
    with tarfile.open(archives[0], "r:gz") as archive:
        members = {
            PurePosixPath(member.name)
            .relative_to(PurePosixPath(member.name).parts[0])
            .as_posix(): member
            for member in archive.getmembers()
        }
        assert not any(".agents" in PurePosixPath(name).parts for name in members)
        packaged_skill = {
            name: member
            for name, member in members.items()
            if name.startswith("skills/audio-cli/") and member.isfile()
        }
        assert packaged_skill.keys() == source_files.keys()
        for name, path in source_files.items():
            stream = archive.extractfile(packaged_skill[name])
            assert stream is not None
            with stream:
                assert stream.read() == path.read_bytes(), name
