# Copyright (c) 2023 Ruiqi Chen
# Email: rchensix at alumni dot stanford dot edu
# MIT License.
# See LICENCE file for full license terms.

import tet_symmetry

import unittest

import numpy as np

class TestTetSymmetry(unittest.TestCase):
    def test_basic(self):
        '''Feed in grid data that is already tetrahedrally symmetric. Create an
        approximation for grid data, evaluate, and see if it matches the input.
        '''
        n = 4
        v = np.arange(-n // 2, n // 2) / n
        x, y, z = np.meshgrid(v, v, v, indexing='ij')
        data = np.cos(2.0 * np.pi * x) + np.cos(2.0 * np.pi * y) + \
               np.cos(2.0 * np.pi * z)
        ts = tet_symmetry.TetSymmetry(data)
        vals = ts.EvaluateNaive(x, y, z)
        self.assertLess(np.linalg.norm(data - vals), 1.0e-6)

    def test_face_normal(self):
        '''Feed in grid data that is already tetrahedrally symmetric. Create an
        approximation for grid data with face normal constraints turned on, and
        evaluate the numerical gradient on the unit tet face.
        '''
        n = 4
        v = np.arange(-n // 2, n // 2) / n
        x, y, z = np.meshgrid(v, v, v, indexing='ij')
        data = np.cos(2.0 * np.pi * x) + np.cos(2.0 * np.pi * y) + \
               np.cos(2.0 * np.pi * z)
        ts = tet_symmetry.TetSymmetry(data, normal_to_face=True)

if __name__ == '__main__':
    unittest.main()