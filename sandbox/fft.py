import numpy as np
import scipy.fft

rng = np.random.default_rng()
for n in range(4, 5):
    data = rng.standard_normal((n, n, n))
    fftn = scipy.fft.fftn(data, norm='forward')
    rfftn = scipy.fft.rfftn(data, norm='forward')
    ifftn = scipy.fft.ifftn(fftn, norm='forward')
    irfftn = scipy.fft.irfftn(rfftn, s=(n, n, n), norm='forward')
    print('n=', n)
    print(fftn.shape)
    print(rfftn.shape)
    print(np.linalg.norm(np.imag(rfftn)))
    print(np.linalg.norm(fftn[:, :, :n//2+1] - rfftn))
    print(np.linalg.norm(ifftn - irfftn))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                val1 = fftn[i, j, k]
                ni = 0 if i == 0 else n - i
                nj = 0 if j == 0 else n - j
                nk = 0 if k == 0 else n - k
                val2 = fftn[ni, nj, nk]
                if (np.linalg.norm(val1 - np.conj(val2)) > 1e-6):
                    print('i: {}, j: {}, ni: {}, nj: {}, val1: {}, val2: {}'.format(i, j, ni, nj, val1, val2))