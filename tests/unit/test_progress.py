import io
import unittest

from apps.common.progress import ConsoleProgress, NullProgress, console


class Terminal(io.StringIO):
    def isatty(self):
        return True


class ConsoleProgressTest(unittest.TestCase):
    def test_non_root_progress_is_a_noop(self):
        prog = console(enabled=False)

        self.assertIsInstance(prog, NullProgress)
        with prog:
            prog.stage("hidden")
            prog({"type": "stage", "message": "hidden"})

    def test_plain_stream_prints_stages_and_iterations(self):
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
        self.assertNotIn("forward", out)

    def test_terminal_updates_solve_phase_in_place(self):
        stream = Terminal()
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


if __name__ == "__main__":
    unittest.main()
