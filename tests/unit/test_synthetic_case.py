from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from femx_cfd import VelocityData
from tools.synthetic.case import (
    _noise_parameters,
    build_synthetic_case,
    load_synthetic_config,
    validate_synthetic_config,
)


CONFIG = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "inverse"
    / "straight_tube"
    / "synthetic.yaml"
)


class SyntheticConfigTest(unittest.TestCase):
    def test_nonroot_rank_only_participates_in_truth_solve(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["solver"]["backend"] = "petsc"
        document["boundary_conditions"][0]["boundary"] = "flow-entry"
        document["boundary_conditions"][1]["boundary"] = "solid-wall"
        document["boundary_conditions"][2]["boundary"] = "flow-exit"
        runtime = Mock()
        runtime.is_root = False
        inlet = object()
        progress = Mock()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "generated"
            with (
                patch(
                    "tools.synthetic.case.parallel.for_case",
                    return_value=runtime,
                ),
                patch("tools.synthetic.case._materialize_mesh") as materialize,
                patch(
                    "tools.synthetic.case.make_inlet",
                    return_value=inlet,
                ) as make_inlet,
                patch(
                    "tools.synthetic.case.solve_truth_observations",
                    return_value=(object(), object(), object()),
                ) as solve,
            ):
                result = build_synthetic_case(
                    document,
                    output,
                    progress=progress,
                    finalize_runtime=True,
                )

            self.assertEqual(result, output.resolve())
            self.assertFalse(output.exists())

        materialize.assert_not_called()
        truth_case = make_inlet.call_args.args[0]
        self.assertEqual(truth_case["control"]["boundary"], "flow-entry")
        self.assertEqual(
            [condition["boundary"] for condition in truth_case["bc"]],
            ["flow-entry", "solid-wall", "flow-exit"],
        )
        self.assertIs(solve.call_args.args[1], inlet)
        self.assertIsNone(solve.call_args.args[2])
        self.assertEqual(runtime.barrier.call_count, 3)
        runtime.finalize.assert_called_once_with()
        runtime.collect.assert_not_called()

    def test_loads_versioned_example_config(self):
        config = load_synthetic_config(CONFIG)

        self.assertEqual(config["format"], "femx-cfd-synthetic")
        self.assertEqual(config["version"], 2)
        self.assertEqual(config["output"]["directory"], str(CONFIG.parent))
        self.assertEqual(config["output"]["every"], 1)
        self.assertEqual(config["mesh"]["length"], 4.0)
        self.assertEqual(
            [item["boundary"] for item in config["boundary_conditions"]],
            ["inlet", "wall", "outlet"],
        )
        inlet = config["boundary_conditions"][0]
        self.assertEqual(inlet["direction"], "auto")
        self.assertEqual(inlet["profile"]["skew"], 0.0)
        self.assertEqual(config["solver"]["backend"], "dense")
        self.assertEqual(config["observations"]["mask"], "none")
        self.assertEqual(config["observations"]["selected_phases"], [1, 2, 3, 4])
        self.assertEqual(config["observations"]["noise"]["seed"], 42)
        self.assertEqual(config["reconstruction"]["kind"], "laplacian")
        self.assertEqual(config["reconstruction"]["temporal_weight"], 0.0)

    def test_accepts_poisson_fourier_reconstruction_without_axial_decay(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["reconstruction"] = {
            "kind": "poisson_fourier",
            "normal": "auto",
            "depth": 0.5,
            "fourier_order": 1,
        }

        config = validate_synthetic_config(document)

        self.assertEqual(config["reconstruction"]["kind"], "poisson_fourier")
        self.assertEqual(config["reconstruction"]["fourier_order"], 1)
        self.assertNotIn("axial_decay", config["reconstruction"])

    def test_accepts_optional_poisson_fourier_axial_decay(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["reconstruction"] = {
            "kind": "poisson_fourier",
            "normal": "auto",
            "depth": 0.5,
            "axial_decay": 0.25,
            "fourier_order": 1,
        }

        config = validate_synthetic_config(document)

        self.assertEqual(config["reconstruction"]["axial_decay"], 0.25)

    def test_accepts_existing_mesh_relative_to_config(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["mesh"] = {"path": "mesh/mesh.msh", "length_unit": "m"}

        config = validate_synthetic_config(document, base_dir=CONFIG.parent)

        self.assertEqual(
            config["mesh"]["path"],
            str(CONFIG.parent / "mesh" / "mesh.msh"),
        )
        self.assertEqual(config["mesh"]["length_unit"], "m")

    def test_rejects_mixed_existing_and_generated_mesh_options(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["mesh"] = {
            "path": "mesh.msh",
            "kind": "straight_tube",
            "length_unit": "m",
        }

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            validate_synthetic_config(document, base_dir=CONFIG.parent)

    def test_rejects_negative_temporal_weight_without_mutating_input(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        original = deepcopy(document)
        document["reconstruction"]["temporal_weight"] = -1.0

        with self.assertRaisesRegex(ValueError, "temporal_weight"):
            validate_synthetic_config(document)

        self.assertEqual(original["reconstruction"]["temporal_weight"], 0.0)

    def test_rejects_unknown_analysis_option(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["analysis"] = {"solver": {"backend": "petsc"}}

        with self.assertRaisesRegex(ValueError, "unknown options: analysis"):
            validate_synthetic_config(document)

    def test_rejects_version_one(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["version"] = 1

        with self.assertRaisesRegex(ValueError, "version must be 2"):
            validate_synthetic_config(document)

    def test_rejects_old_harmonics_key(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        waveform = document["boundary_conditions"][0]["waveform"]
        waveform["harmonics"] = waveform.pop("terms")

        with self.assertRaisesRegex(ValueError, "unknown options: harmonics"):
            validate_synthetic_config(document)

    def test_version_two_requires_explicit_boundary_mapping(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["boundary_conditions"].pop(0)

        with self.assertRaisesRegex(ValueError, "exactly one velocity condition"):
            validate_synthetic_config(document)

        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document.pop("boundary_conditions")

        with self.assertRaisesRegex(ValueError, "boundary_conditions"):
            validate_synthetic_config(document)

    def test_requires_output_every_to_divide_one_period(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["output"]["every"] = 3

        with self.assertRaisesRegex(ValueError, "divide the 4 integration steps"):
            validate_synthetic_config(document)

    def test_resolves_output_directory_relative_to_recipe(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["output"]["directory"] = "generated/case"

        config = validate_synthetic_config(document, base_dir=CONFIG.parent)

        self.assertEqual(
            config["output"]["directory"],
            str((CONFIG.parent / "generated/case").resolve()),
        )

    def test_rejects_empty_output_directory(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["output"]["directory"] = ""

        with self.assertRaisesRegex(ValueError, "output.directory"):
            validate_synthetic_config(document)

    def test_normalizes_explicit_inlet_direction(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["boundary_conditions"][0]["direction"] = [3.0, 4.0]

        config = validate_synthetic_config(document)

        self.assertEqual(
            config["boundary_conditions"][0]["direction"],
            [0.6, 0.8],
        )
        self.assertEqual(
            document["boundary_conditions"][0]["direction"],
            [3.0, 4.0],
        )

    def test_rejects_unknown_top_level_option(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["truth"] = {}

        with self.assertRaisesRegex(ValueError, "unknown options: truth"):
            validate_synthetic_config(document)

    def test_accepts_selected_observation_phases(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["observations"]["selected_phases"] = [2, 4]

        config = validate_synthetic_config(document)

        self.assertEqual(config["observations"]["selected_phases"], [2, 4])

    def test_accepts_mesh_observation_mask(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["observations"]["mask"] = "mesh"

        config = validate_synthetic_config(document)

        self.assertEqual(config["observations"]["mask"], "mesh")

    def test_rejects_unknown_observation_mask(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["observations"]["mask"] = "wall"

        with self.assertRaisesRegex(ValueError, "observations.mask"):
            validate_synthetic_config(document)

    def test_rejects_invalid_selected_observation_phases(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["observations"]["selected_phases"] = [2, 2]

        with self.assertRaisesRegex(ValueError, "unique and increasing"):
            validate_synthetic_config(document)

        document["observations"]["selected_phases"] = [5]
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            validate_synthetic_config(document)

    def test_accepts_linear_snr_instead_of_noise_std(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["observations"]["noise"] = {"snr": 20, "seed": 42}

        config = validate_synthetic_config(document)

        self.assertEqual(config["observations"]["noise"]["snr"], 20.0)
        self.assertEqual(config["observations"]["noise"]["reference"], "rms")
        self.assertNotIn("std", config["observations"]["noise"])

    def test_accepts_maximum_velocity_as_snr_reference(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["observations"]["noise"] = {
            "snr": 20,
            "reference": "max",
            "seed": 42,
        }

        config = validate_synthetic_config(document)

        self.assertEqual(config["observations"]["noise"]["reference"], "max")

    def test_rejects_unknown_snr_reference(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["observations"]["noise"] = {
            "snr": 20,
            "reference": "mean",
            "seed": 42,
        }

        with self.assertRaisesRegex(ValueError, "reference must be 'rms' or 'max'"):
            validate_synthetic_config(document)

    def test_rejects_reference_without_snr(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["observations"]["noise"] = {
            "std": 0.01,
            "reference": "max",
            "seed": 42,
        }

        with self.assertRaisesRegex(ValueError, "reference requires snr"):
            validate_synthetic_config(document)

    def test_rejects_noise_std_and_snr_together(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["observations"]["noise"] = {
            "std": 0.01,
            "snr": 20,
            "seed": 42,
        }

        with self.assertRaisesRegex(ValueError, "either std or snr"):
            validate_synthetic_config(document)

    def test_rejects_nonpositive_noise_snr(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["observations"]["noise"] = {"snr": 0, "seed": 42}

        with self.assertRaisesRegex(ValueError, "observations.noise.snr"):
            validate_synthetic_config(document)

    def test_resolves_noise_std_from_signal_rms_and_snr(self):
        velocity = VelocityData(
            points=np.array([[0.0, 0.0]]),
            times=np.array([0.0, 1.0]),
            vals=np.array([[[3.0, 4.0]], [[0.0, 0.0]]]),
        )

        signal_rms, noise_std = _noise_parameters(
            velocity,
            {"snr": 5.0, "seed": 42},
        )

        self.assertEqual(signal_rms, 2.5)
        self.assertEqual(noise_std, 0.5)

        signal_rms, noise_std = _noise_parameters(
            velocity,
            {"std": 0.1, "seed": 42},
        )

        self.assertEqual(signal_rms, 2.5)
        self.assertEqual(noise_std, 0.1)

    def test_resolves_noise_std_from_truth_maximum_and_snr(self):
        velocity = VelocityData(
            points=np.array([[0.0, 0.0]]),
            times=np.array([0.0]),
            vals=np.array([[[1.0, 0.0]]]),
        )
        model = Mock()
        model.mesh.dimension = 2
        model.mesh.num_nodes = 2
        model.velocity_dofs = np.arange(4)
        trajectory = Mock()
        trajectory.values = np.array(
            [
                [3.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 6.0, 8.0],
            ]
        )

        signal_max, noise_std = _noise_parameters(
            velocity,
            {"snr": 5.0, "reference": "max", "seed": 42},
            model=model,
            trajectory=trajectory,
        )

        self.assertEqual(signal_max, 10.0)
        self.assertEqual(noise_std, 2.0)

    def test_rejects_snr_for_zero_signal(self):
        velocity = VelocityData(
            points=np.array([[0.0, 0.0]]),
            times=np.array([0.0]),
            vals=np.zeros((1, 1, 2)),
        )

        with self.assertRaisesRegex(ValueError, "nonzero signal scale"):
            _noise_parameters(
                velocity,
                {"snr": 20.0, "reference": "rms", "seed": 42},
            )


if __name__ == "__main__":
    unittest.main()
