
import numpy as np

def _fit_plane_svd(P: np.ndarray):
    """
    P: (N,3)
    returns: (c, n) where plane is (x-c)·n=0, |n|=1
    """
    c = P.mean(axis=0)
    X = P - c
    # smallest singular vector = plane normal
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    n = Vt[-1]
    n = n / (np.linalg.norm(n) + 1e-12)
    return c, n

def _plane_basis(n: np.ndarray):
    """
    n: (3,) unit normal
    returns: (u, v) orthonormal basis on plane
    """
    # pick a vector not parallel to n
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return u, v

def _fit_circle_2d_least_squares(xy: np.ndarray):
    """
    xy: (N,2)
    Fit circle in algebraic form: x^2+y^2 + ax + by + c = 0
    returns: center_2d (2,), radius (float)
    """
    x = xy[:, 0]
    y = xy[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x*x + y*y)
    # solve [a,b,c]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_, c = sol
    cx = -a / 2.0
    cy = -b_ / 2.0
    r2 = cx*cx + cy*cy - c
    r = float(np.sqrt(max(r2, 0.0)))
    return np.array([cx, cy], dtype=float), r

def fit_circle_center_3d(points_xyz, return_radius=False):
    """
    points_xyz: list/np.ndarray of shape (N,3), 원 위(또는 근처) 점들
    return: center3d (3,)  [optionally radius]
    """
    P = np.asarray(points_xyz, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 3:
        raise ValueError("points_xyz는 (N,3)이고 N>=3 이어야 합니다.")

    # 1) 평면 피팅
    c_plane, n = _fit_plane_svd(P)

    # 2) 평면 기준축(u,v) 만들기
    u, v = _plane_basis(n)

    # 3) 점들을 평면 좌표계로 투영 (2D)
    X = P - c_plane
    x2 = X @ u
    y2 = X @ v
    xy = np.column_stack([x2, y2])

    # 4) 2D 원 피팅
    center2d, r = _fit_circle_2d_least_squares(xy)

    # 5) 3D 중심으로 복원
    center3d = c_plane + center2d[0] * u + center2d[1] * v

    if return_radius:
        return center3d, r, n  # 중심, 반지름, 평면법선(참고용)
    return center3d
