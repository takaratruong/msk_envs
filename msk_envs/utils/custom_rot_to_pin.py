import numpy as np

# Convert 1dof axis to Pin Joint
axis = "<axis>-0.10501355 -0.17402245 0.97912631999999999</axis>"
n = np.fromstring(axis.split(">")[1].split("<")[0], sep=" ")
n = n / np.linalg.norm(n)

z = np.array([0., 0., 1.])

c = np.dot(z, n)
s = np.sqrt(1 - c ** 2)

# k = z cross n, normalized
k = np.cross(z, n)
k_norm = np.linalg.norm(k)

if k_norm < 1e-10:
    # n is already aligned with z (or anti-aligned), no rotation needed
    R = np.eye(3) if c > 0 else -np.eye(3)
else:
    k = k / k_norm
    kx, ky, kz = k
    one_minus_c = 1 - c

    R = np.array([
        [c + one_minus_c * kx ** 2, one_minus_c * kx * ky - s * kz, one_minus_c * kx * kz + s * ky],
        [one_minus_c * ky * kx + s * kz, c + one_minus_c * ky ** 2, one_minus_c * ky * kz - s * kx],
        [one_minus_c * kz * kx - s * ky, one_minus_c * kz * ky + s * kx, c + one_minus_c * kz ** 2]
    ])

print("R[:,2] should equal n:", R[:, 2])
print("Close to n?", np.allclose(R[:, 2], n))

beta = np.arcsin(R[0, 2])
alpha = np.arctan2(-R[1, 2], R[2, 2])
gamma = np.arctan2(-R[0, 1], R[0, 0])
print("For axis:", n)
print(f"<orientation>{alpha} {beta} {gamma}</orientation>")
print("R[:,2] should equal n:", R[:, 2])

# Gimbal joints: ZXY to XYZ
R = np.array([[0, 1, 0],
              [0, 0, 1],
              [1, 0, 0]], dtype=float)

beta = np.arcsin(R[0, 2])
alpha = np.arctan2(-R[1, 2], R[2, 2])
gamma = np.arctan2(-R[0, 1], R[0, 0])

print()
print("For R:", R)
print(f"<orientation>{alpha} {beta} {gamma}</orientation>")


# Gimbal joints: Z, -X, -Y to XYZ
R = np.array([[ 0, -1,  0],
              [ 0,  0, -1],
              [ 1,  0,  0]], dtype=float)

beta  = np.arcsin(R[0, 2])
alpha = np.arctan2(-R[1, 2], R[2, 2])
gamma = np.arctan2(-R[0, 1], R[0, 0])
print()
print("For R:", R)
print(f"<orientation>{alpha} {beta} {gamma}</orientation>")
