"""Setuptools entry point; dependency metadata is declared in pyproject.toml."""

from collections import defaultdict
from pathlib import Path

from setuptools import setup


def runtime_resources():
    """Keep the runnable examples with their sibling scripts in wheel installs."""
    root = Path(__file__).resolve().parent
    resources = defaultdict(list)
    for folder in ("examples", "recipe", "scripts"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh", ".txt", ".yaml", ".json"}:
                relative = path.relative_to(root)
                resources[f"share/oc-opd/{relative.parent.as_posix()}"].append(relative.as_posix())
    for path in sorted((root / "data").rglob(".gitkeep")):
        relative = path.relative_to(root)
        resources[f"share/oc-opd/{relative.parent.as_posix()}"].append(relative.as_posix())
    resources["share/oc-opd"] = ["NOTICE", *sorted(p.name for p in root.glob("requirements*.txt"))]
    return sorted(resources.items())


if __name__ == "__main__":
    setup(data_files=runtime_resources())
