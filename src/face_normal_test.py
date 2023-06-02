# Copyright (c) 2023 Ruiqi Chen
# Email: rchensix at alumni dot stanford dot edu
# MIT License.
# See LICENCE file for full license terms.

# This file contains the tests that use the normal_to_face=True feature of
# TetSymmetry.

# WARNING: I cannot get cvxpy to play nicely with my other dependencies in my
# Anaconda environment, so this file only contains tests that depend on cvxpy,
# which are only the ones that initialize TetSymmetry with normal_to_face=True.
# To visualize results, the evaluated grids are saved as npy files and
# triangulated with marching cubes in a different test suite that does not
# depend on cvxpy.

import unittest

class TestFaceNorma(unittest.TestCase):
    def test_basic_tet_symmetry(self):
        '''Simple test to ensure TetSymmetry with face normality works.
        '''
        npy_path = '/tmp/basic_tet_symmetry.npy'
        import tet_symmetry
        n = 4
        v = np.arange(-n // 2, n // 2) / n
        x, y, z = np.meshgrid(v, v, v, indexing='ij')
        data = -np.cos(2.0 * np.pi * x) - np.cos(2.0 * np.pi * y) - \
                np.cos(2.0 * np.pi * z)
        ts = tet_symmetry.TetSymmetry(data, normal_to_face=True)
        # Evaluate in the unit cube and use more points so the resulting
        # mesh looks better.
        m = 100
        v = np.linspace(-0.5, 0.5, m)
        xgrid, ygrid, zgrid = np.meshgrid(v, v, v, indexing='ij')
        vals = ts.EvaluateNaive(xgrid, ygrid, zgrid)
        self.assertEqual(vals.shape, xgrid.shape)
        np.save(npy_path, vals)
