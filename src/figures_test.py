# Copyright (c) 2023 Ruiqi Chen
# Email: rchensix at alumni dot stanford dot edu
# MIT License.
# See LICENCE file for full license terms.

# This file contains the tests used to generate the meshes for the figures in
# the manuscript, as well as some miscellaneous meshes for testing.

# WARNING: I cannot get cvxpy to play nicely with my other dependencies in my
# Anaconda environment, so any tests that depend on cvxpy (only the ones that
# initialize TetSymmetry with normal_to_face=True) have been moved to a
# different test file.

import unittest

import numpy as np

import marching_cubes
import mesh_utils
import signed_distance
import tet_symmetry

class TestFigures(unittest.TestCase):
    def test_discretization_level(self):
        '''Test different values of N to see effect on discretization. Also
        test difference between using binary and signed distance field inputs.
        '''
        v, f = mesh_utils.ReadTriangleMesh('src/mesh_inputs/cross_unit_cell.stl')
        min_pt = np.full(3, -0.5)
        n_vals = [32, 64, 128]
        modes = ['binary', 'sdf']
        for n in n_vals:
            max_pt = min_pt + (n - 1) / n
            sdf = signed_distance.ComputeSignedDistance(v, f, min_pt, max_pt,
                                                        np.full(3, n))
            for mode in modes:
                print('Processing N={} mode={}'.format(n, mode))
                if mode == 'binary':
                    data = np.copy(sdf)
                    data[data > 0.0] = 1.0
                    data[data <= 0.0] = -1.0
                else:
                    data = sdf
                ts = tet_symmetry.TetSymmetry(data)
                # Evaluate in the unit cube and use more points so the mesh
                # looks better.
                m = 256
                # Need to add extra layer to make grid periodic for marching cubes.
                vals = np.zeros((m + 1, m + 1, m + 1))
                vals[:-1, :-1, :-1] = np.real(ts.EvaluateUnitCube(m))
                vals[-1, :, :] = vals[0, :, :]
                vals[:, -1, :] = vals[:, 0, :]
                vals[:, :, -1] = vals[:, :, 0]
                voxel_size = 1 / m
                vout, fout = marching_cubes.Triangulate(vals, voxel_size, min_pt,
                                                    pad=True)
                out_path = '/tmp/cross_n{}_{}.ply'.format(n, mode)
                mesh_utils.WriteTriangleMesh(vout, fout, out_path)

if __name__ == '__main__':
    unittest.main()
