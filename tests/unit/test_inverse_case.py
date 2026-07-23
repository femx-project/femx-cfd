"""Tests for portable inverse case loading and validation."""

from pathlib import Path
import unittest

import yaml

from femx_cfd.cases import INVERSE_FORMAT, load_case, snapshot_case
from femx_cfd.cases.inverse import validate_inverse_case


EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "inverse" / "straight_tube"


class InverseCaseTest(unittest.TestCase):
    def test_loads_relative_inputs_for_var(self):
        case = load_case(EXAMPLE / "var" / "case.yaml")

        self.assertEqual(case.format, INVERSE_FORMAT)
        self.assertEqual(case.version, 3)
        self.assertEqual(case.resolved["method"], "var")
        self.assertEqual(case.resolved["control"]["kind"], "vector")
        self.assertTrue(Path(case.resolved["mesh"]["path"]).is_absolute())
        self.assertTrue(Path(case.resolved["observations"]["path"]).is_file())
        self.assertEqual(case.resolved["algorithm"]["optimizer"], "scipy")
        self.assertEqual(
            case.resolved["boundary_conditions"][0],
            {
                "boundary": "wall",
                "kind": "velocity",
                "value": [0.0, 0.0],
            },
        )

    def test_loads_envar_ensemble_options(self):
        case = load_case(EXAMPLE / "envar" / "case.yaml")

        self.assertEqual(case.resolved["method"], "envar")
        self.assertEqual(case.resolved["algorithm"]["outer_iterations"], 2)
        self.assertEqual(case.resolved["output"]["ensemble_basis"], "final")
        self.assertEqual(case.resolved["ensemble"]["workers"], 1)
        modes = case.resolved["ensemble"]["modes"]
        self.assertEqual(modes["spatial"]["uniform"]["kind"], "uniform")
        self.assertEqual(modes["combined"][0]["name"], "flow_mean")
        self.assertEqual(case.resolved["algorithm"]["prior_weight"], 0.25)
        self.assertEqual(
            case.resolved["ensemble"]["profile_scaling"],
            "instantaneous",
        )

    def test_accepts_peak_ensemble_profile_scaling(self):
        document = yaml.safe_load((EXAMPLE / "envar" / "case.yaml").read_text())
        document["ensemble"]["profile_scaling"] = "peak"

        case = validate_inverse_case(document, base_dir=EXAMPLE / "envar")

        self.assertEqual(case["ensemble"]["profile_scaling"], "peak")

    def test_defaults_envar_to_one_outer_iteration(self):
        document = yaml.safe_load((EXAMPLE / "envar" / "case.yaml").read_text())
        document["algorithm"].pop("outer_iterations")

        case = validate_inverse_case(document, base_dir=EXAMPLE / "envar")

        self.assertEqual(case["algorithm"]["outer_iterations"], 1)

    def test_accepts_all_outer_ensemble_basis_output(self):
        document = yaml.safe_load((EXAMPLE / "envar" / "case.yaml").read_text())
        document["output"]["ensemble_basis"] = "all"

        case = validate_inverse_case(document, base_dir=EXAMPLE / "envar")

        self.assertEqual(case["output"]["ensemble_basis"], "all")

    def test_snapshot_records_ensemble_basis_output(self):
        case = load_case(EXAMPLE / "envar" / "case.yaml")

        snapshot = snapshot_case(case, EXAMPLE / "nested" / "result")

        self.assertEqual(snapshot["output"]["ensemble_basis"], "final")

    def test_rejects_unknown_ensemble_basis_output(self):
        document = yaml.safe_load((EXAMPLE / "envar" / "case.yaml").read_text())
        document["output"]["ensemble_basis"] = "none"

        with self.assertRaisesRegex(ValueError, "output.ensemble_basis"):
            validate_inverse_case(document, base_dir=EXAMPLE / "envar")

    def test_rejects_ensemble_basis_output_for_var(self):
        document = yaml.safe_load((EXAMPLE / "var" / "case.yaml").read_text())
        document["output"]["ensemble_basis"] = "all"

        with self.assertRaisesRegex(ValueError, "unknown options"):
            validate_inverse_case(document, base_dir=EXAMPLE / "var")

    def test_rejects_unknown_ensemble_option(self):
        document = yaml.safe_load((EXAMPLE / "envar" / "case.yaml").read_text())
        document["ensemble"]["save_initial_states"] = True

        with self.assertRaisesRegex(ValueError, "unknown options"):
            validate_inverse_case(document, base_dir=EXAMPLE / "envar")

    def test_rejects_var_ensemble_options(self):
        document = yaml.safe_load((EXAMPLE / "var" / "case.yaml").read_text())
        document["ensemble"] = {"workers": 1}

        with self.assertRaisesRegex(ValueError, "only valid for method: envar"):
            validate_inverse_case(document, base_dir=EXAMPLE / "var")

    def test_rejects_var_options_in_envar_algorithm(self):
        document = yaml.safe_load((EXAMPLE / "envar" / "case.yaml").read_text())
        document["algorithm"]["max_iterations"] = 10

        with self.assertRaisesRegex(ValueError, "unknown options: max_iterations"):
            validate_inverse_case(document, base_dir=EXAMPLE / "envar")

    def test_rejects_nonpositive_envar_outer_iterations(self):
        document = yaml.safe_load((EXAMPLE / "envar" / "case.yaml").read_text())
        document["algorithm"]["outer_iterations"] = 0

        with self.assertRaisesRegex(ValueError, "outer_iterations"):
            validate_inverse_case(document, base_dir=EXAMPLE / "envar")

    def test_accepts_relative_observation_time_weights(self):
        document = yaml.safe_load((EXAMPLE / "envar" / "case.yaml").read_text())
        document["observations"]["time_weights"] = [4.0, 1.0]

        case = validate_inverse_case(document, base_dir=EXAMPLE / "envar")

        self.assertEqual(case["observations"]["time_weights"], [4.0, 1.0])

    def test_rejects_nonpositive_observation_time_weight(self):
        document = yaml.safe_load((EXAMPLE / "envar" / "case.yaml").read_text())
        document["observations"]["time_weights"] = [1.0, 0.0]

        with self.assertRaisesRegex(ValueError, r"time_weights\[1\]"):
            validate_inverse_case(document, base_dir=EXAMPLE / "envar")

    def test_rejects_mode_with_unknown_spatial_definition(self):
        document = yaml.safe_load((EXAMPLE / "envar" / "case.yaml").read_text())
        document["ensemble"]["modes"]["combined"][0]["spatial"] = "missing"

        with self.assertRaisesRegex(ValueError, "spatial mode"):
            validate_inverse_case(document, base_dir=EXAMPLE / "envar")

    def test_rejects_unknown_inverse_method(self):
        document = yaml.safe_load((EXAMPLE / "var" / "case.yaml").read_text())
        document["method"] = "other"

        with self.assertRaisesRegex(ValueError, "method"):
            validate_inverse_case(document, base_dir=EXAMPLE / "var")

    def test_rejects_no_slip_boundary_preset(self):
        document = yaml.safe_load((EXAMPLE / "var" / "case.yaml").read_text())
        document["boundary_conditions"][0] = {
            "boundary": "wall",
            "kind": "no_slip",
        }

        with self.assertRaisesRegex(ValueError, "velocity.*pressure"):
            validate_inverse_case(document, base_dir=EXAMPLE / "var")

    def test_accepts_observation_derived_initialization(self):
        document = yaml.safe_load((EXAMPLE / "var" / "case.yaml").read_text())
        document["initialization"]["inlet"] = {
            "source": "observations",
            "projection": {
                "depth": 0.5,
                "axial_decay": 0.25,
                "laplacian_modes": 3,
                "ridge": 0.001,
            },
        }

        case = validate_inverse_case(document, base_dir=EXAMPLE / "var")

        self.assertEqual(case["initialization"]["inlet"]["source"], "observations")
        self.assertEqual(
            case["initialization"]["inlet"]["projection"]["temporal_weight"],
            0.0,
        )
        self.assertEqual(case["control"]["kind"], "vector")

    def test_defaults_current_control_to_vector(self):
        document = yaml.safe_load((EXAMPLE / "var" / "case.yaml").read_text())
        document["control"].pop("kind")

        case = validate_inverse_case(document, base_dir=EXAMPLE / "var")

        self.assertEqual(case["control"], {"boundary": "inlet", "kind": "vector"})

    def test_accepts_optional_normal_control(self):
        document = yaml.safe_load((EXAMPLE / "var" / "case.yaml").read_text())
        document["control"] = {
            "boundary": "inlet",
            "kind": "normal",
            "normal": "auto",
        }
        document["algorithm"]["control_bounds"] = [0.0, 0.2]

        case = validate_inverse_case(document, base_dir=EXAMPLE / "var")

        self.assertEqual(case["control"]["kind"], "normal")
        self.assertEqual(case["control"]["normal"], "auto")

    def test_rejects_normal_field_for_vector_control(self):
        document = yaml.safe_load((EXAMPLE / "var" / "case.yaml").read_text())
        document["control"]["normal"] = "auto"

        with self.assertRaisesRegex(ValueError, "only valid for normal"):
            validate_inverse_case(document, base_dir=EXAMPLE / "var")

    def test_normal_control_keeps_nonnegative_lower_bound(self):
        document = yaml.safe_load((EXAMPLE / "var" / "case.yaml").read_text())
        document["control"] = {"boundary": "inlet", "kind": "normal"}

        with self.assertRaisesRegex(ValueError, "nonnegative"):
            validate_inverse_case(document, base_dir=EXAMPLE / "var")

    def test_migrates_version_1_default_to_normal_control(self):
        document = yaml.safe_load((EXAMPLE / "var" / "case.yaml").read_text())
        document["version"] = 1
        document.pop("output")
        document["control"] = {"boundary": "inlet", "normal": "auto"}
        document["algorithm"]["control_bounds"] = [0.0, 0.2]

        case = validate_inverse_case(document, base_dir=EXAMPLE / "var")

        self.assertEqual(case["version"], 3)
        self.assertEqual(case["control"]["kind"], "normal")
        self.assertEqual(case["control"]["normal"], "auto")

    def test_snapshot_keeps_referenced_paths_valid(self):
        case = load_case(EXAMPLE / "var" / "case.yaml")
        destination = EXAMPLE / "nested" / "result"

        snapshot = snapshot_case(case, destination)
        mesh = (destination / snapshot["mesh"]["path"]).resolve()
        observations = (destination / snapshot["observations"]["path"]).resolve()

        self.assertEqual(mesh, Path(case.resolved["mesh"]["path"]))
        self.assertEqual(
            observations,
            Path(case.resolved["observations"]["path"]),
        )
        self.assertEqual(snapshot["output"]["directory"], ".")
        self.assertEqual(snapshot["output"]["every"], 1)


if __name__ == "__main__":
    unittest.main()
