def vec_yup_zup(v):
    return [v[0], -v[2], v[1]]


def quat_xyzw_to_wxyz(quat_xyzw):
    return quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]
