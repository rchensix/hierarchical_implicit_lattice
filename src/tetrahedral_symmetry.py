# Copyright (c) 2023 Ruiqi Chen
# Email: rchensix at alumni dot stanford dot edu
# MIT License.
# See LICENCE file for full license terms.

# This file contains constants and utility functions that define achiral (Td)
# tetrahedral symmetry.

import numpy as np

# Space group 215 point symmetries
# See http://img.chem.ucl.ac.uk/sgp/large/215az1.htm
kSymmetries = np.array([
    # Identity
    # 1
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]],
    # Rotations by 180 degrees
    # 2
    [[1, 0, 0],
     [0, -1, 0],
     [0, 0, -1]],
    # 3
    [[-1, 0, 0],
     [0, 1, 0],
     [0, 0, -1]],
    # 4
    [[-1, 0, 0],
     [0, -1, 0],
     [0, 0, 1]],
    # Not sure about the rest...
    # 5
    [[0, 0, 1],
     [1, 0, 0],
     [0, 1, 0]],
    # 6
    [[0, 0, -1],
     [-1, 0, 0],
     [0, 1, 0]],
    # 7
    [[0, 0, 1],
     [-1, 0, 0],
     [0, -1, 0]],
    # 8
    [[0, 0, -1],
     [1, 0, 0],
     [0, -1, 0]],
    # 9
    [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0]],
    # 10
    [[0, -1, 0],
     [0, 0, 1],
     [-1, 0, 0]],
    # 11
    [[0, -1, 0],
     [0, 0, -1],
     [1, 0, 0]],
    # 12
    [[0, 1, 0],
     [0, 0, -1],
     [-1, 0, 0]],
    # 13
    [[-1, 0, 0],
     [0, 0, 1],
     [0, -1, 0]],
    # 14
    [[-1, 0, 0],
     [0, 0, -1],
     [0, 1, 0]],
    # 15
    [[1, 0, 0],
     [0, 0, 1],
     [0, 1, 0]],
    # 16
    [[1, 0, 0],
     [0, 0, -1],
     [0, -1, 0]],
    # 17
    [[0, 0, -1],
     [0, -1, 0],
     [1, 0, 0]],
    # 18
    [[0, 0, 1],
     [0, -1, 0],
     [-1, 0, 0]],
    # 19
    [[0, 0, 1],
     [0, 1, 0],
     [1, 0, 0]],
    # 20
    [[0, 0, -1],
     [0, 1, 0],
     [-1, 0, 0]],
    # 21
    [[0, 1, 0],
     [-1, 0, 0],
     [0, 0, -1]],
    # 22
    [[0, -1, 0],
     [1, 0, 0],
     [0, 0, -1]],
    # 23
    [[0, 1, 0],
     [1, 0, 0],
     [0, 0, 1]],
    # 24
    [[0, -1, 0],
     [-1, 0, 0],
     [0, 0, 1]],
], dtype=int)

# This is a regular tetrahedron centered at the origin.
# Vertices are ordered in VTK ordering.
kUnitTetPts = np.array([
    [-1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, 1.0],
    [1.0, -1.0, 1.0],
])

kUnitTetCell = np.array([[0, 1, 2, 3]], dtype=int)

# 3 planes that bound the unique integer lattice points to consider for
# tetrahedral symmetry. All 3 planes go through the origin.
kIntegerLatticePlanes = np.array([
    [0, 1, 1],
    [0, 1, -1],
    [1, -1, 0],
] dtype=int)
