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

def _IsSymmetric(mat: np.ndarray):
    if mat.shape[0] != mat.shape[1]: return False
    n = mat.shape[0]
    kTol = 1e-12
    for i in range(n):
        for j in range(n):
            if np.abs(mat[i][j] - mat[j][i]) > kTol: return False
    return True

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
        self.kTol = 1e-12  # Used to compare against 0
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
        # [-0.5, 0.5)^3, so we do a shift here by 0.5 in every direction.
        xyz += 0.5
        for key, subkey_dict in self.freqs.items():
            # Skip terms with nearly zero coefficients.
            if np.abs(self.basis_coeffs[key]) < self.kTol: continue
            for subkey, num_appearances in subkey_dict.items():
                f = np.array(subkey)
                vals += self.basis_coeffs[key] * self.normalizing_coeffs[key] \
                        * num_appearances * np.exp(2j * np.pi * xyz.dot(f))
        return vals.reshape(x.shape)
    
    def EvaluateUnitCube(self, res: int) -> np.ndarray:
        '''Evaluate the fitted function on the res x res x res grid spanning
        the domain [-0.5, 0.5)^3. For example, the result at index [0, 0, 0]
        comes from evaluating the spatial coordinate x = y = z = -0.5, and the
        result at index [-1, -1, -1] comes from evaluating the spatial
        coordinate x = y = z = -0.5 + (res - 1) / n.

        Args:
            res: int - grid resolution to evaluate at.
        Returns:
            vals: (res, res, res) - evaluated complex values.
        '''
        res_actual = res
        if res < self.n:
            res_actual = int(np.ceil(self.n / res)) * res
        # Create fft grid of size res_actual^3.
        fft = np.zeros((res_actual, res_actual, res_actual),
                       dtype='complex128')
        # Fill fft grid with coefficient values.
        # TODO(rchensix): Cache this fft grid since it only needs to be shifted
        # for different res_actual values.
        for key, subkey_dict in self.freqs.items():
            # Skip terms with nearly zero coefficients.
            if np.abs(self.basis_coeffs[key]) < self.kTol: continue
            for subkey, num_appearances in subkey_dict.items():
                fx, fy, fz = subkey
                idx0 = fx if fx >= 0 else fx + res_actual
                idx1 = fy if fy >= 0 else fy + res_actual
                idx2 = fz if fz >= 0 else fz + res_actual
                fft[idx0, idx1, idx2] = self.basis_coeffs[key] * \
                                        self.normalizing_coeffs[key] * \
                                        num_appearances
        return scipy.fft.ifftn(fft, norm='forward')

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
        if self.normal_to_face: self._ComputeCoeffsNormalToFace()
    
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
        # Get affine transform that projects points onto all four tetrahedron
        # faces of the unit tetrahedron.
        # WARNING: This is hardcoded based on how kUnitTetPts is defined.
        # If you change kUnitTetPts, you need to manually change this too.
        normals = np.array([
            [1, 1, 1],
            [-1, -1, 1],
            [1, -1, -1],
            [-1, 1, -1],
        ], dtype=int)
        proj_mats_transp = np.array([
            [[1, 0, -1],
             [0, 1, -1],
             [0, 0, 0]],
            [[1, 0, 1],
             [0, 1, 1],
             [0, 0, 0]],
            [[1, 0, 1],
             [0, 1, -1],
             [0, 0, 0]],
            [[1, 0, -1],
             [0, 1, 1],
             [0, 0, 0]],
        ], dtype=int)
        # TODO(rchensix): clean this up.
        # WARNING: scipy.fft computes coefficients assuming the unit cube is in
        # [0, 1)^3, so we have to shift the planes that we enforce normality.
        offsets = np.array([
            [0, 0, 1],
            # [0, 0, 0.5],
            # [0, 0, -0.5],
            # [0, 0, -0.5],
        ])
        for i in range(1):
            p_t = proj_mats_transp[i]
            b = offsets[i]
            n = normals[i]
            # Collect all equality constraint equations.
            # There is one equation per unique p_t @ f_subkey.
            # The unknowns are the coefficients for each unique f.
            constraints = dict()
            for key, subfreq_dict in self.freqs.items():
                # Skip the constant term since it doesn't contribute to the
                # gradient.
                if key == (0, 0, 0): continue
                for subkey, num_appearances in subfreq_dict.items():
                    f_subkey = np.array(subkey)
                    f_proj = p_t.dot(f_subkey)
                    b_term = np.exp(2j * np.pi * b.dot(f_subkey))
                    grad_term = 2j * np.pi * n.dot(f_subkey)
                    coeff = self.normalizing_coeffs[key] * num_appearances * \
                            b_term * grad_term
                    projkey = tuple(f_proj)
                    if projkey in constraints:
                        if key in constraints[projkey]:
                            constraints[projkey][key] += coeff
                        else:
                            constraints[projkey][key] = coeff
                    else:
                        constraints[projkey] = {
                            key: coeff
                        }
        # Prune out any constraint equations where every entry is 0.
        self.constraints = dict()
        for projkey, coeffs_dict in constraints.items():
            is_nonzero = False
            for _, coeff in coeffs_dict.items():
                if np.abs(coeff) > self.kTol:
                    is_nonzero = True
                    break
            if is_nonzero:
                self.constraints[projkey] = coeffs_dict
        self._OptimizeCoeffs()
    
    def _OptimizeCoeffs(self):
        # Put the cvxpy import here since this is the only method that needs it.
        # This makes it easier to run other methods in TetSymmetry since cvxpy
        # does not get along with many geometry packages on my computer.
        import cvxpy as cp
        key_to_index = dict()
        num_keys = 0
        for key in self.freqs:
            # Skip the constant term since it's known and won't be changed.
            if key == (0, 0, 0): continue
            key_to_index[key] = num_keys
            num_keys += 1

        # Set up q, p, a, b matrices.
        # See https://www.cvxpy.org/examples/basic/quadratic_program.html
        p = np.eye(num_keys)
        q = np.zeros(num_keys, dtype='complex128')
        for key, index in key_to_index.items():
            q[index] = self.basis_coeffs[key]
        num_constraints = len(self.constraints)
        a = np.zeros((num_constraints, num_keys), dtype='complex128')
        b = np.zeros(num_constraints)
        row = 0
        for _, coeffs_dict in self.constraints.items():
            for key, coeff in coeffs_dict.items():
                index = key_to_index[key]
                a[row, index] = coeff
            row += 1

        # Define and solve the CVXPY problem.
        x = cp.Variable(num_keys, complex=True)
        obj = cp.Minimize(cp.quad_form(x, p) - 2 * cp.real(q.T @ x))
        prob = cp.Problem(obj, [a @ x == b])
        prob.solve(verbose=True)
        # Print result.
        print("\nThe optimal value is", prob.value)
        sol = x.value
        print("A solution x is", sol)

        # Set new coefficients
        for key, index in key_to_index.items():
            self.basis_coeffs[key] = sol[index]
