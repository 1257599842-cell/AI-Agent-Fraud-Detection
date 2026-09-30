"""Keep the published Python environments explicit and mutually compatible."""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pins(path):
    result = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([\w.-]+)==([^\s;]+)", line)
        if match is None:
            raise AssertionError(f"Expected a version pin in {path.name}: {line}")
        name = re.sub(r"[-_.]+", "-", match[1]).lower()
        if name in result:
            raise AssertionError(f"Duplicate dependency: {name}")
        result[name] = match[2]
    return result


class TestDependencyContract(unittest.TestCase):
    def test_core_environment_has_no_notebook_or_browser_stack(self):
        core = pins(ROOT / "requirements.txt")
        for name in ("jupyterlab", "jupyter-server", "seaborn", "playwright"):
            self.assertNotIn(name, core)
        for name in ("lightgbm", "duckdb", "fastapi", "httpx", "pandas"):
            self.assertIn(name, core)
        for name, version in pins(ROOT / "requirements/core.in").items():
            self.assertEqual(core[name], version)

    def test_extended_environments_preserve_core_versions(self):
        core = pins(ROOT / "requirements.txt")
        for filename, added in (("browser.txt", "playwright"),
                                ("notebooks.txt", "jupyterlab")):
            extended = pins(ROOT / "requirements" / filename)
            self.assertIn(added, extended)
            for name, version in core.items():
                self.assertEqual(extended[name], version, f"{filename}: {name}")

    def test_ci_and_container_use_the_documented_environment(self):
        workflow = (ROOT / ".github/workflows/verify.yml").read_text()
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn('python-version: "3.13"', workflow)
        self.assertIn("pip install -r requirements/browser.txt", workflow)
        self.assertIn("python -m pip check", workflow)
        self.assertIn("FROM python:3.13-slim", dockerfile)
