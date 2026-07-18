"""Tests for the bundled portable example cases."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from femx_cfd.cases import load_case
from femx_cfd.cli import run_cli


class ExampleRunnerTest(unittest.TestCase):
    def test_bundled_cases_use_the_public_formats(self):
        root = Path(__file__).resolve().parents[2] / "femx_cfd" / "examples"

        forward = load_case(root / "forward/straight_tube/case.yaml")
        var = load_case(root / "inverse/straight_tube/var.yaml")
        envar = load_case(root / "inverse/straight_tube/envar.yaml")

        self.assertEqual(forward.format, "femx-cfd-forward")
        self.assertEqual(var.resolved["method"], "var")
        self.assertEqual(envar.resolved["method"], "envar")

    def test_example_dispatches_through_the_generic_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("femx_cfd.cli.run_case") as run:
                code = run_cli(["example", "var", "--output", directory])

        self.assertEqual(code, 0)
        self.assertEqual(Path(run.call_args.args[0]).name, "var.yaml")
        self.assertEqual(run.call_args.kwargs["progress_kind"], "console")


if __name__ == "__main__":
    unittest.main()
