import meshio
import numpy as np

kUnitTetPts = np.array([
    [-0.5, -0.5, -0.5],
    [0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
])

kUnitTetCell = np.array([[0, 1, 2, 3]], dtype=int)

meshio.write_points_cells('test_data/unit_tet.vtu', kUnitTetPts, [('tetra', kUnitTetCell)])