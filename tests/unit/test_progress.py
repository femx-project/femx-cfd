import io
import json
import unittest

import numpy as np

from femx_cfd.runtime.progress import (
    ConsoleProgress,
    JsonProgress,
    NullProgress,
)


class ConsoleProgressTest(unittest.TestCase):
    def test_non_root_progress_is_a_noop(self):
        prog = NullProgress()

        self.assertIsInstance(prog, NullProgress)
        with prog:
            prog.stage("hidden")
            prog({"type": "stage", "message": "hidden"})

    def test_plain_stream_updates_solves_in_place(self):
        stream = io.StringIO()
        prog = ConsoleProgress(stream)

        prog({"type": "stage", "message": "optimization: tao / petsc"})
        prog(
            {
                "type": "solve",
                "phase": "forward-begin",
                "step": 0,
                "total": 3,
            }
        )
        prog(
            {
                "type": "optimizer",
                "iteration": 2,
                "max_iterations": 10,
                "objective": 1.25,
                "gradient_norm": 0.5,
            }
        )

        out = stream.getvalue()
        self.assertIn("optimization: tao / petsc", out)
        self.assertIn("iter    2/10", out)
        self.assertIn("objective 1.250000e+00", out)
        self.assertIn("\r\033[Kforward     0/3", out)
        self.assertIn("\n\033[1A", out)

    def test_close_clears_in_place_solve(self):
        stream = io.StringIO()
        prog = ConsoleProgress(stream)

        prog(
            {
                "type": "solve",
                "phase": "adjoint-step",
                "step": 2,
                "total": 4,
            }
        )
        prog.close()

        self.assertIn("\r\033[Kadjoint", stream.getvalue())
        self.assertIn("2/4", stream.getvalue())

    def test_ensemble_members_report_independent_forward_steps(self):
        stream = io.StringIO()
        prog = ConsoleProgress(stream)

        prog(
            {
                "type": "ensemble_solve",
                "stage": "spin-up",
                "member": "spin-up flow_mean_x",
                "phase": "forward",
                "step": 2,
                "total": 10,
            }
        )
        prog(
            {
                "type": "ensemble_solve",
                "stage": "spin-up",
                "member": "spin-up skew_mean_y",
                "phase": "forward",
                "step": 5,
                "total": 10,
            }
        )

        out = stream.getvalue()
        self.assertIn("spin-up flow_mean_x", out)
        self.assertIn("spin-up skew_mean_y", out)
        self.assertIn("2/10", out)
        self.assertIn("5/10", out)

    def test_scipy_evaluations_are_not_labeled_as_iterations(self):
        stream = io.StringIO()
        prog = ConsoleProgress(stream)

        prog(
            {
                "type": "optimizer",
                "iteration": 5,
                "max_iterations": 3,
                "iteration_kind": "evaluation",
                "objective": 1.25,
                "gradient_norm": 0.5,
            }
        )

        self.assertIn("eval    5", stream.getvalue())
        self.assertNotIn("5/3", stream.getvalue())


class JsonProgressTest(unittest.TestCase):
    def test_writes_stage_and_solver_events_as_json_lines(self):
        stream = io.StringIO()
        prog = JsonProgress(stream)

        prog.stage("forward: solving")
        prog(
            {
                "type": "solve",
                "step": np.int64(1),
                "total": np.int64(2),
            }
        )

        events = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(
            events,
            [
                {"type": "stage", "message": "forward: solving"},
                {"type": "solve", "step": 1, "total": 2},
            ],
        )

    def test_writes_numpy_arrays_in_optimizer_events(self):
        stream = io.StringIO()
        prog = JsonProgress(stream)

        prog({"type": "optimizer", "param": np.asarray([1.0, 2.0])})

        event = json.loads(stream.getvalue())
        self.assertEqual(event["param"], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
