# Copyright (c) 2023 Ruiqi Chen
# Email: rchensix at alumni dot stanford dot edu
# MIT License.
# See LICENCE file for full license terms.

# This file computes signed distance voxel grids from triangle meshes.

import igl
import numpy as np

def ComputeSignedDistance(v: np.ndarray, f: np.ndarray, min_pt: np.ndarray,
                          max_pt: np.ndarray, num_grid_pts: np.ndarray) \
                              -> np.ndarray:
    '''Computes the signed distance of a triangle mesh (v, f) at voxel grid
    points defined by min_pt to max_pt, inclusive. The number of grid points is
    given by num_grid_pts.

    Args:
        v: (N, 3) - point array.
        f: (M, 3) - triangle array.
        min_pt: (3,) - minimum point of voxel grid.
        max_pt: (3,) - maximum point of voxel grid.
        num_grid_pts: (3,) - integer vector of number of grid points in x, y,
            and z directions.
    Returns:
        sdf: (num_grid_pts[0], num_grid_pts[1], num_grid_pts[2]) - signed
            distance values at every grid point.
    '''
    xvec = np.linspace(min_pt[0], max_pt[0], num_grid_pts[0])
    yvec = np.linspace(min_pt[1], max_pt[1], num_grid_pts[1])
    zvec = np.linspace(min_pt[2], max_pt[2], num_grid_pts[2])
    xgrid, ygrid, zgrid = np.meshgrid(xvec, yvec, zvec, indexing='ij')
    num_pts = np.prod(num_grid_pts)
    query_pts = np.zeros((num_pts, 3))
    query_pts[:, 0] = xgrid.squeeze()
    query_pts[:, 1] = ygrid.squeeze()
    query_pts[:, 2] = zgrid.squeeze()
    sdf, _, _ = igl.signed_distance(query_pts, v, f)
    return sdf.reshape(xgrid.shape)
