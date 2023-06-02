# Copyright (c) 2023 Ruiqi Chen
# Email: rchensix at alumni dot stanford dot edu
# MIT License.
# See LICENCE file for full license terms.

import unittest

import numpy as np

import marching_cubes
import mesh_utils

class TestMarchingCubes(unittest.TestCase):
    def test_sphere(self):
        '''Test sphere centered at origin.
        '''
        n = 100
        domain = np.linspace(-1.0, 1.0, n)
        x, y, z = np.meshgrid(domain, domain, domain, indexing='ij')
        r = 0.5
        grid = np.sqrt(x**2 + y**2 + z**2) - r
        voxel_size = (domain[-1] - domain[0]) / (n - 1)
        min_pt = np.full(3, domain[0])
        v, f = marching_cubes.Triangulate(grid, voxel_size, min_pt)
        mesh_utils.WriteTriangleMesh(v, f, '/tmp/sphere.ply')
    
    def test_padding(self):
        '''Test gyroid with large value padding to close side walls and make
        watertight.
        '''
        n = 100
        domain = np.linspace(0.0, 1.0, n)
        x, y, z = np.meshgrid(domain, domain, domain, indexing='ij')
        grid = np.cos(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y) + \
               np.cos(2.0 * np.pi * y) * np.sin(2.0 * np.pi * z) + \
               np.cos(2.0 * np.pi * z) * np.sin(2.0 * np.pi * x)
        c = 0.2
        grid = grid**2 - c**2
        voxel_size = (domain[-1] - domain[0]) / (n - 1)
        min_pt = np.full(3, domain[0])
        v, f = marching_cubes.Triangulate(grid, voxel_size, min_pt, pad=True)
        mesh_utils.WriteTriangleMesh(v, f, '/tmp/gyroid_padded.ply')

if __name__ == '__main__':
    unittest.main()