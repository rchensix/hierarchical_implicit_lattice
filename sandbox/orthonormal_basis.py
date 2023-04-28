import numpy as np

pts = np.array([
    [-0.5, -0.5, -0.5],
    [0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
])

# n dot (r - r0) = 0
# n dot r - n dot r0 = 0
# nx x + ny y + nz z = n dot r0
# z = (n dot r0 - nx x - ny y) / nz
# z = -nx / nz x - ny / nz y + n dot r0 / nz

# 0 3 2
n = np.cross(pts[3] - pts[0], pts[2] - pts[0])
print(n)
print(-n[0] / n[2], -n[1] / n[2], 0, n.dot(pts[0]) / n[2])

# 0 1 3
n = np.cross(pts[1] - pts[0], pts[3] - pts[0])
print(n)
print(-n[0] / n[2], -n[1] / n[2], 0, n.dot(pts[0]) / n[2])

# 0 2 1
n = np.cross(pts[2] - pts[0], pts[1] - pts[0])
print(n)
print(-n[0] / n[2], -n[1] / n[2], 0, n.dot(pts[0]) / n[2])

m = np.array([[1, 0, -1],
             [0, 1, -1],
             [0, 0, 0]])
print(m.dot(np.array([1, 1, 0])))
