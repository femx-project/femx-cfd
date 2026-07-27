"""Tests for the bundled portable example cases."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from femx_cfd.cases import load_case
from femx_cfd.cli import _example_cases, run_cli


class ExampleRunnerTest(unittest.TestCase):
    def test_bundled_cases_use_the_public_formats(self):
        cases = _example_cases()

        forward = load_case(cases["forward"])
        var = load_case(cases["var"])
        envar = load_case(cases["envar"])

        documents = {
            name: yaml.safe_load(path.read_text(encoding="utf-8"))
            for name, path in cases.items()
        }
        for document in documents.values():
            self.assertEqual(document["solver"]["platform"], "cpu")
            self.assertEqual(document["solver"]["backend"], "petsc")
            self.assertEqual(
                {condition["kind"] for condition in document["boundary_conditions"]},
                {"pressure", "velocity"},
            )

        self.assertEqual(forward.format, "femx-cfd-forward")
        self.assertEqual(documents["forward"]["version"], 3)
        self.assertEqual(forward.resolved["solver"]["platform"], "cpu")
        self.assertEqual(forward.resolved["solver"]["backend"], "petsc")
        self.assertEqual(documents["var"]["version"], 3)
        self.assertEqual(var.resolved["method"], "var")
        self.assertEqual(var.resolved["control"]["kind"], "vector")
        self.assertEqual(var.resolved["solver"]["platform"], "cpu")
        self.assertEqual(var.resolved["solver"]["backend"], "petsc")
        self.assertEqual(documents["envar"]["version"], 3)
        self.assertEqual(envar.resolved["method"], "envar")
        self.assertEqual(envar.resolved["control"]["kind"], "vector")
        self.assertEqual(envar.resolved["solver"]["platform"], "cpu")
        self.assertEqual(envar.resolved["solver"]["backend"], "petsc")

    def test_example_dispatches_through_the_generic_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("femx_cfd.cli.run_case") as run:
                code = run_cli(["example", "var", "--output", directory])

        self.assertEqual(code, 0)
        case = Path(run.call_args.args[0])
        self.assertEqual(case.name, "case.yaml")
        self.assertEqual(case.parent.name, "var")
        self.assertEqual(run.call_args.kwargs["progress_kind"], "console")
        self.assertTrue(run.call_args.kwargs["finalize_runtime"])


if __name__ == "__main__":
    unittest.main()
