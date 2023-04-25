import numpy as np

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
])

terms = dict()
freq = np.array([3, 3, 3])
for symmetry in kSymmetries:
    term = tuple(symmetry @ freq)
    if term in terms:
        terms[term] += 1
    else:
        terms[term] = 1
print(terms)
print(len(terms))

kIntegerLatticePlanes = np.array([
    [0, 1, 1],
    [0, 1, -1],
    [1, -1, 0],
], dtype=int)

pts = list()
for i in range(10):
    for j in range(10):
        for k in range(10):
            if j + k < 0: continue
            if j - k < 0: continue
            if i - j < 0: continue
            pts.append((i, j, k))
pts = np.array(pts)
import matplotlib.pyplot as plt
# ax = plt.figure().add_subplot(projection='3d')
plt.plot(pts[:, 0], pts[:, 1])
print(pts[:, 0])
print(pts[:, 1])
plt.show()