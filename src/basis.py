# Copyright (c) 2023 Ruiqi Chen
# Email: rchensix at alumni dot stanford dot edu
# MIT License.
# See LICENCE file for full license terms.

# This file contains functions for constructing Fourier basis functions with
# tetrahedral symmetry.

import numpy as np

import tetrahedral_symmetry

def _Coeff(f: np.ndarray) -> float:
    '''Computes "K", the normalizing coefficient in front of a basis function.

    Args:
        f: (3,) - integer frequency 3-vector.
    ReturnsL
        k: float - normalizinng coefficient.
    '''
    # There's probably more efficient ways to do this but this is easier to
    # understand.
    freq_count = dict()
    for m in tetrahedral_symmetry.kSymmetries:
        freq = m.transpose() @ f
        key = tuple(freq)
        if key in freq_count:
            freq_count[key] += 1
        else:
            freq_count[key] = 1
    total = 0
    for _, v in freq_count.items():
        total += v**2
    return np.sqrt(total)
