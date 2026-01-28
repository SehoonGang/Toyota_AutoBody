import numpy as np
import open3d as o3d
from itertools import permutations


def estimate_rigid_transform_3d(src_pts: np.ndarray, dst_pts: np.ndarray):
    """Kabsch: dst ~= R @ src + t (scale=1)"""
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)
    assert src_pts.shape == dst_pts.shape and src_pts.shape[1] == 3

    src_c = src_pts.mean(axis=0)
    dst_c = dst_pts.mean(axis=0)

    X = src_pts - src_c
    Y = dst_pts - dst_c

    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # reflection fix
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = dst_c - R @ src_c
    return R, t


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rms_error(src_pts: np.ndarray, dst_pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> float:
    pred = (R @ src_pts.T).T + t[None, :]
    err = np.linalg.norm(pred - dst_pts, axis=1)
    return float(np.sqrt(np.mean(err * err)))


def find_best_transform_order_invariant(src_pts: np.ndarray, dst_pts: np.ndarray):
    """
    dst 순서가 섞여있어도 최적 매칭(순열 탐색)으로 src->dst 변환을 구함.
    반환: T(4x4), best_perm(list), best_rms(float), dst_perm(array)
    """
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)
    assert src_pts.shape == dst_pts.shape
    n = src_pts.shape[0]

    best_rms = np.inf
    best_T = None
    best_perm = None
    best_dst_perm = None

    for perm in permutations(range(n)):
        dst_perm = dst_pts[list(perm), :]
        R, t = estimate_rigid_transform_3d(src_pts, dst_perm)
        e = rms_error(src_pts, dst_perm, R, t)
        if e < best_rms:
            best_rms = e
            best_T = make_T(R, t)
            best_perm = list(perm)
            best_dst_perm = dst_perm

    return best_T, best_perm, best_rms, best_dst_perm


def apply_T_to_points(points_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """(N,3) points에 4x4 transform 적용"""
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_xyz must be (N,3). got {pts.shape}")

    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])          # (N,4)
    out_h = (T @ pts_h.T).T                 # (N,4)
    return out_h[:, :3]


def save_two_sets_as_ply_pointcloud(weldings_xyz: np.ndarray,
                                    src_xyz: np.ndarray,
                                    out_ply: str,
                                    welding_color=(1.0, 0.2, 0.2),
                                    src_color=(0.0, 0.0, 0.0)):
    w = np.asarray(weldings_xyz, dtype=np.float64)
    s = np.asarray(src_xyz, dtype=np.float64)

    if w.size == 0 and s.size == 0:
        raise ValueError("Both point sets are empty")
    if w.ndim != 2 or w.shape[1] != 3:
        raise ValueError(f"weldings_xyz must be (N,3). got {w.shape}")
    if s.ndim != 2 or s.shape[1] != 3:
        raise ValueError(f"src_xyz must be (N,3). got {s.shape}")

    points = np.vstack([w, s])  # (Nw+Ns, 3)

    colors_w = np.tile(np.array(welding_color, dtype=np.float64), (w.shape[0], 1))
    colors_s = np.tile(np.array(src_color, dtype=np.float64), (s.shape[0], 1))
    colors = np.vstack([colors_w, colors_s])  # (Nw+Ns, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    ok = o3d.io.write_point_cloud(out_ply, pcd, write_ascii=False)
    if not ok:
        raise RuntimeError(f"Failed to write: {out_ply}")

    print(f"[OK] saved: {out_ply}  (weldings={w.shape[0]}, src_trans={s.shape[0]})")


if __name__ == "__main__":
    src = np.array([
        [510.12, 448.33, 1241.80],
        [817.61, 447.69, 1236.33],
        [1228.10, 381.45, 1212.50],
        [768.06, 443.70, 1262.38],
        [796.11, 443.71, 1208.86],
    ], dtype=np.float64)

    dst = np.array([
        [-279.23, -776.86, -549.16],
        [   0.53, -721.61, -552.32],
        [ -22.64, -777.32, -553.56],
        [  24.15, -747.10, -548.27],
        [ 429.50, -688.07, -611.54],
    ], dtype=np.float64)

    cad_weldings = np.array([
        [476.73, 517.53, 1172.96],
        [483, 486, 1188],
        [477.16, 482.02, 1299.12],
        [467.78, 534.01, 1317.12],
        [475.73, 552.98, 1325.45],
        [605.66, 473.81, 1197.60],
        [589.59, 445.79, 1208.93],
        [604.73, 454.42, 1236.20],
        [604.41, 448.45, 1271.75],
        [605.56, 464.05, 1293.05],

        [637.56, 522.66, 1314.71],
        [656.64, 484.94, 1174.90],
        [700.72, 459, 1201.25],
        [700.79, 451.06, 1236.30],
        [701, 445, 1271.35],
        [702.36, 469.25, 1294],
        [729, 441, 1208],
        [744, 441.69, 1273.09],
        [747, 469, 1294],
        [773.86, 441.72, 1208.73],

        [792, 441.69, 1273],
        [792.36, 469, 1294],
        [818.36, 441.7, 1208.73],
        [836.74, 463, 1197.60],
        [836.37, 450.89, 1236.33],
        [836, 444.88, 1271.45],
        [837.36, 469, 1294],
        [931, 470, 1197.60],
        [931.95, 450.87, 1236.33],
        [932, 444.87, 1271.45],

        [932, 469, 1294],
        [954.55, 440.81, 1208.69],
        [962, 441.67, 1273],
        [971.68, 463.88, 1293],
        [1014.14, 474.62, 1183.17],
        [1019.91, 432.64, 1202.58],
        [1019.91, 434, 1272.75],
        [1019.69, 457.43, 1292.57],
        [1048, 474, 1180],
        [1078.39, 418.39, 1237.27],

        [1085, 449, 1293],
        [1094.91, 464.66, 1180.40],
        [1092, 429.49, 1180.40],
        [1101, 410, 1201],
        [1128.93, 451.75, 1180.40],
        [1135.57, 417, 1180.40],
        [1193.57, 403.96, 1180.40],
        [1258.41, 394.76, 1171.50],
        [1141.72, 397.85, 1201.20],
        [1182.50, 387.34, 1201.20]
    ], dtype=np.float64)

    # cad_scale = 1 / 1.0055786
    # src *= cad_scale
    # cad_weldings *= cad_scale

    # ✅ 순서 무관 자동 매칭으로 최적 T 구하기
    T, perm, best_rms, dst_perm = find_best_transform_order_invariant(src, dst)
    print("[Best RMS]", best_rms)
    print("[Best dst perm indices]", perm)
    print("T=\n", T)

    # ✅ src 변환 결과 (빠져있던 부분!)
    src_trans = apply_T_to_points(src, T)

    # (선택) 5점 정합 결과 확인
    per_point_err = np.linalg.norm(src_trans - dst_perm, axis=1)
    print("[5-point errors]", per_point_err)

    # cad_weldings 변환 결과
    weldings_trans = apply_T_to_points(cad_weldings, T)

    # ✅ 둘을 한 PLY로 저장 (cad_weldings=빨강, src_trans=검정)
    out_ply = r"C:\Users\SehoonKang\Desktop\cad_weldings_plus_srcTrans.ply"
    save_two_sets_as_ply_pointcloud(
        weldings_xyz=weldings_trans,
        src_xyz=src_trans,
        out_ply=out_ply,
        welding_color=(1.0, 0.2, 0.2),  # 빨강
        src_color=(0.0, 0.0, 0.0)       # 검정
    )
