"""The offline report family owns projection policy and cannot execute media or models."""

import ast
from pathlib import Path

from audio_cli.pipeline import compare_reports, summarize_report

ROOT = Path(__file__).resolve().parents[2] / "src/audio_cli/pipeline"
MODULES = {
    "__init__",
    "comparison",
    "compatibility",
    "evidence",
    "loading",
    "matching",
    "metrics",
    "summary",
}


def test_report_family_inventory_and_public_facade():
    assert {path.stem for path in (ROOT / "reports").glob("*.py")} == MODULES
    assert compare_reports.__module__ == "audio_cli.pipeline.reports.comparison"
    assert summarize_report.__module__ == "audio_cli.pipeline.reports.summary"
    assert not (ROOT / "report_compare.py").exists()
    assert not (ROOT / "report_metrics.py").exists()


def test_report_policy_depends_only_on_its_family_error_type_and_file_identity():
    allowed = MODULES | {"models", "media"}
    external = {
        "__future__",
        "collections",
        "fractions",
        "json",
        "math",
        "os",
        "pathlib",
        "re",
        "typing",
    }
    for path in (ROOT / "reports").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] in external for alias in node.names), path
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    assert node.module in allowed, path
                    if node.module == "media":
                        assert [alias.name for alias in node.names] == [
                            "file_identity_from_descriptor"
                        ]
                    if node.module == "models":
                        assert [alias.name for alias in node.names] == ["PipelineError"]
                else:
                    assert node.module.split(".")[0] in external, path
