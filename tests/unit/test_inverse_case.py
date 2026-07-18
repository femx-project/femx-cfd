"""Tests for portable inverse case loading and validation."""

from pathlib import Path
import unittest

import yaml

from femx_cfd.cases import INVERSE_FORMAT, load_case, snapshot_case
from femx_cfd.cases.inverse import validate_inverse_case


EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "inverse" / "straight_tube"


class InverseCaseTest(unittest.TestCase):
    def test_loads_relative_inputs_for_var(self):
        case = load_case(EXAMPLE / "var.yaml")

        self.assertEqual(case.format, INVERSE_FORMAT)
        self.assertEqual(case.version, 1)
        self.assertEqual(case.resolved["method"], "var")
        self.assertTrue(Path(case.resolved["mesh"]["path"]).is_absolute())
        self.assertTrue(Path(case.resolved["observations"]["path"]).is_file())
        self.assertEqual(case.resolved["algorithm"]["optimizer"], "scipy")

    def test_loads_envar_ensemble_options(self):
        case = load_case(EXAMPLE / "envar.yaml")

        self.assertEqual(case.resolved["method"], "envar")
        self.assertEqual(case.resolved["ensemble"]["workers"], 1)
        self.assertIn("flow_mean", case.resolved["ensemble"]["mode_scales"])

    def test_rejects_unknown_inverse_method(self):
        document = yaml.safe_load((EXAMPLE / "var.yaml").read_text())
        document["method"] = "other"

        with self.assertRaisesRegex(ValueError, "method"):
            validate_inverse_case(document, base_dir=EXAMPLE)

    def test_accepts_observation_derived_initialization(self):
        document = yaml.safe_load((EXAMPLE / "var.yaml").read_text())
        document["initialization"]["inlet"] = {
            "source": "observations",
            "projection": {
                "depth": 0.5,
                "axial_decay": 0.25,
                "laplacian_modes": 3,
                "ridge": 0.001,
            },
        }

        case = validate_inverse_case(document, base_dir=EXAMPLE)

        self.assertEqual(case["initialization"]["inlet"]["source"], "observations")

    def test_snapshot_keeps_referenced_paths_valid(self):
        case = load_case(EXAMPLE / "var.yaml")
        destination = EXAMPLE / "nested" / "result"

        snapshot = snapshot_case(case, destination)
        mesh = (destination / snapshot["mesh"]["path"]).resolve()
        observations = (destination / snapshot["observations"]["path"]).resolve()

        self.assertEqual(mesh, Path(case.resolved["mesh"]["path"]))
        self.assertEqual(
            observations,
            Path(case.resolved["observations"]["path"]),
        )


if __name__ == "__main__":
    unittest.main()
