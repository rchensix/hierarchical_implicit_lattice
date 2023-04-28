# Copyright (c) 2023 Ruiqi Chen
# Email: rchensix at alumni dot stanford dot edu
# MIT License.
# See LICENCE file for full license terms.

# This file contains functions for constructing Fourier basis functions with
# tetrahedral symmetry.

from typing import Tuple, Union

import numpy as np
import skimage.measure

def Triangulate(grid: np.ndarray, voxel_size: Union[float, int, np.ndarray],
                min_pt: np.ndarray, isolevel: float=0.0,
                pad: bool=False) -> \
                Tuple[np.ndarray, np.ndarray]:
    '''Triangulate a voxel grid using the convention that inside voxels are
    negative. The variable grid defines the corners of the voxels. If
    grid.shape[0] == M, then it means there are M - 1 voxels along the x
    direction. The world space coordinate of the corner point grid[0, 0, 0] is
    located at min_pt and the size of every voxel is defined by voxel_size.

    Args:
        grid: (M, N, P) - voxel grid points. There are M grid points along the
            x direction, N along y, and P along z. This corresponds to M - 1
            voxels along x, N - 1 voxels along y, and P - 1 voxels along z.
            M, N, and P must all be >= 2.
        voxel_size: float or (3,) - length of voxel in x, y, z directions.
        min_pt: (3,) - location of corner point grid[0, 0, 0].
        isolevel: float - what isolevel to triangulate. Default is 0.
        pad: bool - if True, pads border with large-valued voxels.
            Typically used to create watertight meshes.
    '''
    assert len(grid.shape) == 3, 'grid must be of shape (M, N, P)'
    m, n, p = grid.shape
    assert m >= 2, 'M must be >= 2'
    assert n >= 2, 'N must be >= 2'
    assert p >= 2, 'P must be >= 2'
    if isinstance(voxel_size, float) or isinstance(voxel_size, int):
        assert voxel_size > 0, 'voxel_size must be > 0'
        voxel_size_real = np.full(3, voxel_size)
    else:
        assert voxel_size.shape == (3,), 'voxel_size must be of shape (3,)'
        for i in range(3):
            val = voxel_size[i]
            assert val > 0.0, 'voxel_size[{}] must be > 0'.format(val)
        voxel_size_real = voxel_size
    
    if pad:
        kPad = 1
        kBigBoi = 1.0e6
        pad_shape = (m + 2 * kPad, n + 2 * kPad, p + 2 * kPad)
        grid_real = np.full(pad_shape, kBigBoi)
        grid_real[kPad:-kPad, kPad:-kPad, kPad:-kPad] = grid
    else:
        grid_real = grid
    v, f, _, _ = skimage.measure.marching_cubes(grid_real, level=isolevel,
                                                spacing=voxel_size_real)
    v += min_pt
    if pad:
        v -= kPad * voxel_size_real 
    return v, f
