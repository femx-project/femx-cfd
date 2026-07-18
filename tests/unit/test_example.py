"""Tests for the bundled portable example cases."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from femx_cfd.cases import load_case
from femx_cfd.cli import _example_cases, run_cli


class ExampleRunnerTest(unittest.TestCase):
    def test_bundled_cases_use_the_public_formats(self):
        cases = _example_cases()

        forward = load_case(cases["forward"])
        var = load_case(cases["var"])
        envar = load_case(cases["envar"])

        self.assertEqual(forward.format, "femx-cfd-forward")
        self.assertEqual(forward.resolved["solver"]["backend"], "petsc")
        self.assertEqual(var.resolved["method"], "var")
        self.assertEqual(var.resolved["solver"]["backend"], "petsc")
        self.assertEqual(envar.resolved["method"], "envar")
        self.assertEqual(envar.resolved["solver"]["backend"], "petsc")

    def test_example_dispatches_through_the_generic_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("femx_cfd.cli.run_case") as run:
                code = run_cli(["example", "var", "--output", directory])

        self.assertEqual(code, 0)
        self.assertEqual(Path(run.call_args.args[0]).name, "var.yaml")
        self.assertEqual(run.call_args.kwargs["progress_kind"], "console")
        self.assertTrue(run.call_args.kwargs["finalize_runtime"])


if __name__ == "__main__":
    unittest.main()
