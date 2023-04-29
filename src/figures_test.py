# Copyright (c) 2023 Ruiqi Chen
# Email: rchensix at alumni dot stanford dot edu
# MIT License.
# See LICENCE file for full license terms.

# This file contains the tests used to generate the meshes for the figures in
# the manuscript, as well as some miscellaneous meshes for testing.

# WARNING: I cannot get cvxpy to play nicely with my other dependencies in my
# Anaconda environment, so to run this test, you'll have to do something silly:
# 1. Activate the environment with cvxpy.
# 2. Run using the command "python figures_test.py --mode=fit". This will run
#    the tetrahedral symmetry fitting portions of the test and dump out the
#    voxel grids as .npy files in the mesh_outputs folder.
# 3. Re-run using the command "python figures_test.py --mode=tri". This will
#    run the triangulation portions of the test and dump out the PLY files in
#    mesh_outputs folder.

# TODO(rchen): enable
# import argparse
import unittest

import numpy as np

class TestFigures(unittest.TestCase):
    def test_basic_tet_symmetry(self):
        '''Simple test to ensure TetSymmetry works.
        '''
        npy_path = 'mesh_outputs/basic_tet_symmetry.npy'
        if mode == 'fit':
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
        elif mode == 'tri':
            import marching_cubes
            import mesh_utils
            import signed_distance
            vals = np.load(npy_path)
            m = vals.shape[0]
            voxel_size = 1 / (m - 1)
            min_pt = np.full(3, -0.5)
            v, f = marching_cubes.Triangulate(vals, voxel_size, min_pt, pad=True)
            out_path = 'mesh_outputs/basic_tet_symmetry.ply'
            mesh_utils.WriteTriangleMesh(v, f, out_path)

if __name__ == '__main__':
    # TODO(rchen): use argparse
    # mode = 'fit'
    mode = 'tri'
    unittest.main()