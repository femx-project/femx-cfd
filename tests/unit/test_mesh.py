"""Tests for synthetic mesh generation without requiring Gmsh."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


_MODULE_PATH = Path(__file__).parents[2] / "femx_cfd" / "mesh" / "straight.py"
_SPEC = importlib.util.spec_from_file_location("straight_mesh", _MODULE_PATH)
straight_mesh = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(straight_mesh)
write_straight_tube = straight_mesh.write_straight_tube


class _FakeMesh:
    def __init__(self):
        self.size = None
        self.dimension = None

    def setSize(self, entities, size):
        self.size = (entities, size)

    def generate(self, dimension):
        self.dimension = dimension


class _FakeOcc:
    def addRectangle(self, *args):
        self.rectangle = args
        return 20

    def synchronize(self):
        self.synchronized = True


class _FakeModel:
    def __init__(self):
        self.occ = _FakeOcc()
        self.mesh = _FakeMesh()
        self.groups = []
        self.names = []

    def add(self, name):
        self.name = name

    def remove(self):
        self.removed = True

    def getBoundary(self, entities, oriented=False):
        return [(1, 10), (1, 11), (1, 12), (1, 13)]

    def getBoundingBox(self, dimension, tag):
        boxes = {
            10: (-4.0, -0.5, 0.0, -4.0, 0.5, 0.0),
            11: (4.0, -0.5, 0.0, 4.0, 0.5, 0.0),
            12: (-4.0, -0.5, 0.0, 4.0, -0.5, 0.0),
            13: (-4.0, 0.5, 0.0, 4.0, 0.5, 0.0),
        }
        return boxes[tag]

    def addPhysicalGroup(self, dimension, entities, tag):
        self.groups.append((dimension, entities, tag))

    def setPhysicalName(self, dimension, tag, name):
        self.names.append((dimension, tag, name))

    def getEntities(self, dimension):
        return [(0, 1), (0, 2), (0, 3), (0, 4)]


class _FakeGmsh(types.ModuleType):
    def __init__(self):
        super().__init__("gmsh")
        self.model = _FakeModel()
        self.option = types.SimpleNamespace(setNumber=self._set_option)
        self.options = []
        self.initialized = False
        self.finalized = False

    def _set_option(self, name, value):
        self.options.append((name, value))

    def isInitialized(self):
        return self.initialized

    def initialize(self):
        self.initialized = True

    def finalize(self):
        self.finalized = True

    def write(self, path):
        Path(path).write_text("mesh", encoding="utf-8")


class StraightTubeMeshTest(unittest.TestCase):
    def test_writes_named_physical_groups(self):
        gmsh = _FakeGmsh()
        previous = sys.modules.get("gmsh")
        sys.modules["gmsh"] = gmsh
        try:
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "tube.msh"
                write_straight_tube(
                    output,
                    length=8.0,
                    diameter=1.0,
                    cell_size=0.05,
                )
                self.assertTrue(output.exists())
        finally:
            if previous is None:
                del sys.modules["gmsh"]
            else:
                sys.modules["gmsh"] = previous

        self.assertEqual(gmsh.model.mesh.dimension, 2)
        self.assertEqual(
            gmsh.model.groups,
            [(2, [20], 3), (1, [10], 4), (1, [11], 5), (1, [12, 13], 6)],
        )
        self.assertEqual(
            gmsh.model.names,
            [(2, 3, "fluid"), (1, 4, "inlet"), (1, 5, "outlet"), (1, 6, "wall")],
        )
        self.assertTrue(gmsh.model.removed)
        self.assertTrue(gmsh.finalized)

    def test_rejects_nonpositive_dimensions_without_gmsh(self):
        with self.assertRaises(ValueError):
            write_straight_tube(
                "unused.msh", length=0, diameter=1, cell_size=0.05
            )
