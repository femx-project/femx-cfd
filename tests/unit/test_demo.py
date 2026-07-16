"""Tests for the one-command demo runner."""

import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

from apps.demo import main as demo


class DemoRunnerTest(unittest.TestCase):
    def test_runs_var_stages_in_order(self):
        output = io.StringIO()
        with mock.patch.object(demo.subprocess, "run") as run:
            run.return_value.returncode = 0
            with redirect_stdout(output):
                self.assertEqual(demo.var_main(), 0)

        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands), 5)
        self.assertEqual(commands[0][2:5], ["apps.mesh.main", "--config-name", "demo/mesh"])
        self.assertEqual(commands[-1][2:5], ["apps.var.main", "--config-name", "demo/var"])
        text = output.getvalue()
        self.assertIn("femx-var-demo  [1/5]  mesh", text)
        self.assertIn("Completed: mesh", text)
        self.assertIn("femx-var-demo  [5/5]  Var", text)
        self.assertIn("Completed: Var", text)
        self.assertIn("femx-var-demo complete", text)
        self.assertIn("\n\n", text)

    def test_runs_envar_as_the_selected_analysis(self):
        output = io.StringIO()
        with mock.patch.object(demo.subprocess, "run") as run:
            run.return_value.returncode = 0
            with redirect_stdout(output):
                self.assertEqual(demo.envar_main(), 0)

        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands), 5)
        self.assertEqual(
            commands[-1][2:5],
            ["apps.envar.main", "--config-name", "demo/envar"],
        )
        self.assertIn("femx-envar-demo  [5/5]  EnVar", output.getvalue())

    def test_stops_after_a_failed_stage(self):
        output = io.StringIO()
        with mock.patch.object(demo.subprocess, "run") as run:
            run.side_effect = [
                mock.Mock(returncode=0),
                mock.Mock(returncode=7),
            ]
            with redirect_stdout(output):
                self.assertEqual(demo.envar_main(), 7)

        self.assertEqual(run.call_count, 2)
        self.assertIn(
            "femx-envar-demo: truth inlet failed (exit code 7)",
            output.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
