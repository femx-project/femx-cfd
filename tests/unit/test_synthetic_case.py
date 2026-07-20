from copy import deepcopy
from pathlib import Path
import unittest

import yaml

from tools.synthetic.case import (
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
    def test_loads_versioned_example_config(self):
        config = load_synthetic_config(CONFIG)

        self.assertEqual(config["format"], "femx-cfd-synthetic")
        self.assertEqual(config["mesh"]["length"], 4.0)
        self.assertEqual(config["truth"]["direction"], "auto")
        self.assertEqual(config["truth"]["profile"]["skew"], 0.0)
        self.assertEqual(config["observations"]["noise"]["seed"], 42)
        self.assertEqual(config["reconstruction"]["temporal_weight"], 0.0)

    def test_rejects_negative_temporal_weight_without_mutating_input(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        original = deepcopy(document)
        document["reconstruction"]["temporal_weight"] = -1.0

        with self.assertRaisesRegex(ValueError, "temporal_weight"):
            validate_synthetic_config(document)

        self.assertEqual(original["reconstruction"]["temporal_weight"], 0.0)

    def test_rejects_analysis_settings_in_synthetic_recipe(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["analysis"] = {"solver": {"backend": "petsc"}}

        with self.assertRaisesRegex(ValueError, "independent Var/EnVar"):
            validate_synthetic_config(document)

    def test_normalizes_explicit_truth_direction(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["truth"]["direction"] = [3.0, 4.0]

        config = validate_synthetic_config(document)

        self.assertEqual(config["truth"]["direction"], [0.6, 0.8])
        self.assertEqual(document["truth"]["direction"], [3.0, 4.0])

    def test_rejects_obsolete_truth_normal(self):
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        document["truth"]["normal"] = "auto"

        with self.assertRaisesRegex(ValueError, "truth.direction"):
            validate_synthetic_config(document)


if __name__ == "__main__":
    unittest.main()
