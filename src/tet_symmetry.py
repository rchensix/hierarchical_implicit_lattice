# Copyright (c) 2023 Ruiqi Chen
# Email: rchensix at alumni dot stanford dot edu
# MIT License.
# See LICENCE file for full license terms.

# This file contains functions for constructing Fourier basis functions with
# tetrahedral symmetry.

from typing import Tuple

import numpy as np
import scipy.fft

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
    [-0.5, -0.5, -0.5],
    [0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
])

kUnitTetCell = np.array([[0, 1, 2, 3]], dtype=int)

# Outward normal vector of 3 planes that bound the unique integer lattice
# points to consider for tetrahedral symmetry. All 3 planes go through the
# origin.
kIntegerLatticePlanes = np.array([
    [0, -1, -1],
    [0, -1, 1],
    [-1, 1, 0],
], dtype=int)

def _IsInsideIntegerLatticePlanes(fx: int, fy: int, fz: int) -> bool:
    '''Returns True if (fx, fy, fz) lies on or inside the tetrahedron bounded
    by kIntegerLatticePlanes.

    Args:
        fx: int - integer frequency in x direction.
        fy: int - integer frequency in y direction.
        fz: int - integer frequency in z direction.
    Returns:
        is_inside: bool - returns True if (fx, fy, fz) lies on or inside
            kIntegerLatticePlanes.
    '''
    for plane in kIntegerLatticePlanes:
        nx, ny, nz = plane
        if fx * nx + fy * ny + fz * nz > 0: return False
    return True

class TetSymmetry:
    def __init__(self, data: np.ndarray, normal_to_face: bool=False,
                 copy_data: bool=True):
        '''This class fits a set of achiral tetrahedrally symmetric (Td) basis
        functions to input grid data.

        Args:
            data: (N, N, N) - array of data to approximate. N must be > 2.
                Data is assumed to be from the domain [-0.5, 0.5) for x, y, z
                directions. For example, if N = 4, then data[0, 0, 0] is the
                value at (-0.5, -0.5, -0.5) and data[-1, -1, -1] is the value
                at (0.25, 0.25, 0.25).
            normal_to_face: bool - if True, adjusts weights such the gradient
                of the approximation is normal to the walls of the unit
                tetrahedron when evaluated on the walls. See kUnitTetPts for
                definition of unit tetrahedron.
                TODO(rchensix): Link to paper section explaining what
                normal_to_face does.
            copy_data: bool - if True, copies data and stores in this class.
                If False, data is not copied and user MUST ensure that data
                passed in is not changed.
        '''
        assert len(data.shape) == 3, 'data must be of shape (N, N, N)'
        n = data.shape[0]
        assert data.shape[1] == n and data.shape[2] == n, \
            'data must be of shape (N, N, N)'
        assert n > 2, 'cannot construct approximation with N={}'.format(n)
        self.n = n
        self.data = np.copy(data) if copy_data else data
        self.normal_to_face = normal_to_face
        self._ComputeCoeffs()

    def EvaluateNaive(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) \
                      -> np.ndarray:
        '''Evaluate the fitted function on points specified by x, y, z.
        The inputs x, y, z should be generated using np.meshgrid with
        indexing='ij'.

        NOTE: This function is naive in that it brute force evaluates each term
        and loops through all terms. There are probably more efficient ways of
        doing this, such as using scipy.fft.ifftn.

        Args:
            x: (P, Q, R) - float array of x values.
            y: (P, Q, R) - float array of y values.
            z: (P, Q, R) - float array of z values.
        Returns:
            vals: (P, Q, R) - evaluated complex values.
        '''
        assert len(x.shape) == 3, 'x, y, z must be of shape (P, Q, R)'
        assert x.shape == y.shape and x.shape == z.shape, \
            'x, y, z must all be the same shape'
        p, q, r = x.shape
        vals = np.zeros(p * q * r, dtype='complex128')
        xyz = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1),
                              z.reshape(-1, 1)), axis=1)
        # NOTE: scipy.fft assumes input was discretized in the domain [0, 1]^3.
        # However, we assume the input was discretized in the domain
        # [-0.5, 0.5]^3, so we do a shift here by 0.5 in every direction.
        xyz += 0.5
        for key, subkey_dict in self.freqs.items():
            for subkey, num_appearances in subkey_dict.items():
                f = np.array(subkey)
                vals += self.basis_coeffs[key] * self.normalizing_coeffs[key] \
                        * num_appearances * np.exp(2j * np.pi * xyz.dot(f))
        return vals.reshape(x.shape)

    def _ComputeCoeffs(self):
        '''Computes coefficients of tetrahedral symmetry.
        '''
        # Compute FFT of input data.
        # NOTE: The norm='forward' part is necessary to get the 1/N scaling.
        # See https://www.johndcook.com/blog/2021/03/20/fourier-series-fft/
        # NOTE: In practice, all signed distance inputs should be real valued,
        # so scipy.fft.rfftn would be more space saving.
        self.fftn = scipy.fft.fftn(self.data, norm='forward')
        # Max frequency to compute is Nyquist frequency.
        self.max_f = self.n // 2 if self.n % 2 == 1 else self.n // 2 - 1
        # Compute unique (fx, fy, fz) frequency triplets as well as their
        # transformations by the point symmetry group.
        self.freqs = dict()
        for fx in range(self.max_f + 1):
            for fy in range(self.max_f + 1):
                for fz in range(self.max_f + 1):
                    if not _IsInsideIntegerLatticePlanes(fx, fy, fz): continue
                    key = (fx, fy, fz)
                    assert key not in self.freqs
                    self.freqs[key] = dict()
                    for m in kSymmetries:
                        subkey = tuple(m.transpose() @ np.array(key))
                        if subkey in self.freqs[key]:
                            self.freqs[key][subkey] += 1
                        else:
                            self.freqs[key][subkey] = 1
        # Compute coefficient for each unique frequency triplet as well as
        # normalizing coefficient out front to ensure the basis functions are
        # orthonormal.
        self.basis_coeffs = dict()
        self.normalizing_coeffs = dict()
        for key, subfreq_dict in self.freqs.items():
            normalizing_coeff = 0
            basis_coeff_sum = 0
            for subkey, num_appearances in subfreq_dict.items():
                # Get the FFT coefficient corresponding to subkey.
                basis_coeff_sum += num_appearances * self._GetCoeff(subkey)
                normalizing_coeff += num_appearances**2
            self.normalizing_coeffs[key] = 1.0 / np.sqrt(normalizing_coeff)
            self.basis_coeffs[key] = basis_coeff_sum * \
                                     self.normalizing_coeffs[key]
        # Maybe modify the coefficients if normal_to_face=True.
        if self.normal_to_face:
            self.basis_coeffs = self._ComputeCoeffsNormalToFace()
    
    def _GetCoeff(self, f: Tuple[int, int, int]) -> float:
        '''Get FFT coefficient at (fx, fy, fz).

        Args:
            f: Tuple[int, int, int] - frequency triplet (fx, fy, fz)
        Returns:
            coeff: float - complex-valued coefficient.
        '''
        fx, fy, fz = f
        idx0 = fx if fx >= 0 else fx + self.n
        idx1 = fy if fy >= 0 else fy + self.n
        idx2 = fz if fz >= 0 else fz + self.n
        return self.fftn[idx0, idx1, idx2]
    
    def _ComputeCoeffsNormalToFace(self):
        raise NotImplementedError('normal_to_face=True not supported yet')
