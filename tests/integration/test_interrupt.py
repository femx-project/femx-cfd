import signal
import subprocess
import sys
import textwrap
import time
import unittest
from pathlib import Path

import femx


ROOT = Path(__file__).resolve().parents[2]


class InterruptTest(unittest.TestCase):
    @unittest.skipUnless(
        "petsc" in femx.solver_backends(),
        "femx was built without PETSc/TAO",
    )
    def test_sigint_stops_tao(self):
        code = textwrap.dedent(
            """
            import numpy as np
            import femx

            from femx_cfd import Var
            from tests.integration.rectangle_demo import make_problem, truth_parameter

            prob, op = make_problem()
            truth = prob.solve(truth_parameter(prob), backend="petsc")
            obs = femx.GaussianObservation(op, values=op.predict(truth))
            var = Var(prob, obs, backend="petsc")
            print("ready", flush=True)
            var.solve(
                np.full(prob.num_param, 1.0e-12),
                opt="tao",
                bnds=[(0.0, 0.3)] * prob.num_param,
                max_itrs=10000,
                grad_tol=0.0,
            )
            print("completed", flush=True)
            """
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            self.assertEqual(proc.stdout.readline().strip(), "ready")
            time.sleep(0.05)
            proc.send_signal(signal.SIGINT)
            output, _ = proc.communicate(timeout=5.0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("KeyboardInterrupt", output)
        self.assertNotIn("completed", output)


if __name__ == "__main__":
    unittest.main()
