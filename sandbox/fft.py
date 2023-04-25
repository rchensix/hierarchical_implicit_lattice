import numpy as np
import scipy.fft

print('HELLO')
n = 4
data = np.sin(np.arange(n) * np.pi * 2 / n)
print(scipy.fft.fft(data, norm='forward'))
print(scipy.fft.rfft(data, norm='forward'))