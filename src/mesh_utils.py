# Copyright (c) 2023 rchensix.
# MIT License.
# See LICENCE file for full license terms.

# This file contains various utility functions for processing meshes.

from typing import List, Tuple, Union

import meshio
import numpy as np
import trimesh

def ReadTriangleMesh(path: str) -> Tuple[np.ndarray, np.ndarray]:
    '''Reads triangle mesh from path using meshio library.

    Args:
        path: str - path to triangle mesh file.
    Returns:
        v: (N, 3) - point array.
        f: (M, 3) - triangle array.
    '''
    mesh = meshio.read(path)
    assert mesh.cells[0][0] == 'triangle', \
        'ReadTriangleMesh only supports triangles.'
    return mesh.points, mesh.cells[0][1]

def WriteTriangleMesh(points: np.ndarray, cells: np.ndarray, out_path: str):
    '''Writes triangle mesh to out_path using meshio library.

    Args:
        points: (N, 3) - point array.
        cells: (M, 3) - triangle array.
        out_path: path to write to.
    Returns:
        None
    '''
    assert points.shape[1] == 3, 'points must be shape (N, 3)'
    assert cells.shape[1] == 3, 'cells must be shape (N, 3)'
    cells_actual = [('triangle', cells)]
    meshio.write_points_cells(out_path, points, cells_actual, binary=True)

def CombineTriangleMesh(meshes: List[Tuple[np.ndarray, np.ndarray]]) \
        -> Tuple[np.ndarray, np.ndarray]:
    '''Crudely combine list of triangle meshes into one triangle mesh.

    Args:
        meshes: [(v0, f0), (v1, f1), ...] - list of meshes to combine.
    Returns:
        v: (N, 3) - combined point array.
        f: (M, 3) - combined triangle array.
    '''
    # Determine num points and cells ahead of time
    num_meshes = len(meshes)
    num_pts = np.zeros(num_meshes, dtype=int)
    num_cells = np.zeros(num_meshes, dtype=int)
    for i, mesh in enumerate(meshes):
        pts, cells = mesh
        num_pts[i] = pts.shape[0]
        num_cells[i] = cells.shape[0]
    num_pts_total = np.sum(num_pts)
    num_cells_total = np.sum(num_cells)
    # Build new mesh
    pts_new = np.zeros((num_pts_total, 3), dtype=float)
    cells_new = np.zeros((num_cells_total, 3), dtype=int)
    num_pts_cumulative = np.cumsum(num_pts)
    num_cells_cumulative = np.cumsum(num_cells)
    for i, mesh in enumerate(meshes):
        pts, cells = mesh
        if i == 0:
            pt_idx_prev = 0
            cell_idx_prev = 0
        else:
            pt_idx_prev = num_pts_cumulative[i - 1]
            cell_idx_prev = num_cells_cumulative[i - 1]
        pt_idx_next = num_pts_cumulative[i]
        cell_idx_next = num_cells_cumulative[i]
        pts_new[pt_idx_prev:pt_idx_next, :] = pts
        cells_new[cell_idx_prev:cell_idx_next, :] = cells + pt_idx_prev
    return pts_new, cells_new

def ReadTetrahedronMesh(path: str) -> Tuple[np.ndarray, np.ndarray]:
    '''Reads tetrahedron mesh from path using meshio library.

    Args:
        path: str - path to tetrahedron mesh file.
    Returns:
        v: (N, 3) - point array.
        f: (M, 4) - tetrahedron array.
    '''
    mesh = meshio.read(path)
    assert mesh.cells[0][0] == 'tetra', \
        'ReadTetrahedronMesh only supports tetrahedron.'
    return mesh.points, mesh.cells[0][1]

def WriteTetrahedronMesh(points: np.ndarray, cells: np.ndarray, out_path: str):
    '''Writes tetrahedron mesh to out_path using meshio library.

    Args:
        points: (N, 3) - point array.
        cells: (M, 4) - tetrahedron array.
        out_path: path to write to.
    Returns:
        None
    '''
    assert points.shape[1] == 3, 'points must be shape (N, 3)'
    assert cells.shape[1] == 4, 'cells must be shape (N, 4)'
    cells_actual = [('tetra', cells.astype(int))]
    meshio.write_points_cells(out_path, points, cells_actual)

def ReadHexahedronMesh(path: str) -> Tuple[np.ndarray, np.ndarray]:
    '''Reads hexahedron mesh from path using meshio library.

    Args:
        path: str - path to hexahedron mesh file.
    Returns:
        v: (N, 3) - point array.
        f: (M, 8) - hexahedron array.
    '''
    mesh = meshio.read(path)
    assert mesh.cells[0][0] == 'hexahedron', \
        'ReadHexahedronMesh only supports hexahedron.'
    return mesh.points, mesh.cells[0][1]

def WriteHexahedronMesh(points: np.ndarray, cells: np.ndarray, out_path: str):
    '''Writes hexahedron mesh to out_path using meshio library.

    Args:
        points: (N, 3) - point array.
        cells: (M, 8) - hexahedron array.
        out_path: path to write to.
    Returns:
        None
    '''
    assert points.shape[1] == 3, 'points must be shape (N, 3)'
    assert cells.shape[1] == 8, 'cells must be shape (N, 8)'
    cells_actual = [('hexahedron', cells.astype(int))]
    meshio.write_points_cells(out_path, points, cells_actual)

def CombineHexahedronMesh(meshes: List[Tuple[np.ndarray, np.ndarray]]) \
        -> Tuple[np.ndarray, np.ndarray]:
    '''Crudely combine list of hexahedron meshes into one triangle mesh.

    Args:
        meshes: [(v0, f0), (v1, f1), ...] - list of meshes to combine.
    Returns:
        v: (N, 3) - combined point array.
        f: (M, 8) - combined hexahedron array.
    '''
    # Determine num points and cells ahead of time
    num_meshes = len(meshes)
    num_pts = np.zeros(num_meshes, dtype=int)
    num_cells = np.zeros(num_meshes, dtype=int)
    for i, mesh in enumerate(meshes):
        pts, cells = mesh
        num_pts[i] = pts.shape[0]
        num_cells[i] = cells.shape[0]
    num_pts_total = np.sum(num_pts)
    num_cells_total = np.sum(num_cells)
    # Build new mesh
    pts_new = np.zeros((num_pts_total, 3), dtype=float)
    cells_new = np.zeros((num_cells_total, 8), dtype=int)
    num_pts_cumulative = np.cumsum(num_pts)
    num_cells_cumulative = np.cumsum(num_cells)
    for i, mesh in enumerate(meshes):
        pts, cells = mesh
        if i == 0:
            pt_idx_prev = 0
            cell_idx_prev = 0
        else:
            pt_idx_prev = num_pts_cumulative[i - 1]
            cell_idx_prev = num_cells_cumulative[i - 1]
        pt_idx_next = num_pts_cumulative[i]
        cell_idx_next = num_cells_cumulative[i]
        pts_new[pt_idx_prev:pt_idx_next, :] = pts
        cells_new[cell_idx_prev:cell_idx_next, :] = cells + pt_idx_prev
    return pts_new, cells_new

def Rotate(points: np.ndarray, axis: np.ndarray, angle_deg: float,
           center: Union[None, np.ndarray]=None) -> np.ndarray:
    '''Rotates points about the given axis by angle_deg. Axis is set at center.
    If center is not specified, the center is assumed to be the origin.

    Args:
        points: (N, 3) - point array.
        axis: (3,) - axis of rotation. Can be non-normalized.
        angle_deg: float - angle in degrees to rotate about axis according to
            right hand rule.
        center: (3,) or None - center of rotation. If None, the origin is used.
    Returns:
        rotated_points: (N, 3) - rotated point array.
    '''
    assert points.shape[1] == 3, 'points must be shape (N, 3)'
    assert axis.shape == (3,), 'axis must be shape (3,)'
    if center is None:
        center = np.zeros(3)
    else:
        assert center.shape == (3,), 'center must be shape (3,)'
    # Normalize axis
    axis_n = axis/np.linalg.norm(axis)
    # Offset points by center
    pts = points - center
    # Rotate points about origin using Rodrigues formula
    # See https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    angle_rad = angle_deg*np.pi/180
    rotated = pts*np.cos(angle_rad) + np.cross(axis_n, pts)*np.sin(angle_rad) + \
              axis_n*np.dot(pts, axis_n).reshape((-1, 1))*(1 - np.cos(angle_rad))
    # Undo offset
    return rotated + center

def IsPointInsideTriangleMesh(query_points: np.ndarray, mesh_path: str) \
        -> np.ndarray:
    '''Returns bool array where true indicates a given point is inside triangle
    mesh.

    Args:
        query_pts: (N, 3) - points to query inside/outside status.
        mesh_path: str - path to triangle mesh.
    Returns:
        containment_status: (N,) - bool array of inside/outside status.
    '''
    mesh = trimesh.load_mesh(mesh_path)
    return mesh.contains(query_points)
