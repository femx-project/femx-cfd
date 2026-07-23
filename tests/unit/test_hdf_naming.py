"""HDF5 naming convention tests."""

from pathlib import Path
import re
import unittest

import h5py


EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
LOWER_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
STEP_NAME = re.compile(r"^step_[0-9]{6}$")


class HdfNamingTest(unittest.TestCase):
    def test_example_hdf5_items_and_attributes_use_lower_snake_case(self):
        paths = sorted(EXAMPLES.rglob("*.h5"))
        self.assertTrue(paths)

        for path in paths:
            with self.subTest(path=path), h5py.File(path) as h5:
                self._assert_attribute_names(h5, "/")

                def check(name, item):
                    for component in name.split("/"):
                        self.assertRegex(component, LOWER_SNAKE_CASE)
                    self._assert_attribute_names(item, name)

                h5.visititems(check)
                if "data" in h5:
                    for name in h5["data"]:
                        self.assertRegex(name, STEP_NAME)

    def _assert_attribute_names(self, item, path):
        for name in item.attrs:
            with self.subTest(path=path, attribute=name):
                self.assertRegex(name, LOWER_SNAKE_CASE)


if __name__ == "__main__":
    unittest.main()
