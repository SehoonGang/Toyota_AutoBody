import os
import numpy as np
import open3d as o3d

def load_asc_points(path, xyz_cols=(0, 1, 2), skiprows=0):
    pts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i < skiprows:
                continue
            s = line.strip()
            if not s:
                continue
            parts = s.replace(",", " ").split()
            if len(parts) <= max(xyz_cols):
                continue
            try:
                x = float(parts[xyz_cols[0]])
                y = float(parts[xyz_cols[1]])
                z = float(parts[xyz_cols[2]])
                pts.append((x, y, z))
            except ValueError:
                continue

    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        raise ValueError("ASC에서 XYZ 포인트를 못 읽었습니다. (헤더/컬럼/구분자 확인)")
    return pts

def apply_depth_colormap(pcd: o3d.geometry.PointCloud,
                         mode="z",               # "z" or "radius"
                         clip_percent=(2, 98),    # 이상치 잘라내기 (보기 좋아짐)
                         invert=False):
    pts = np.asarray(pcd.points)
    if mode == "z":
        depth = pts[:, 2].copy()
    elif mode == "radius":
        depth = np.linalg.norm(pts, axis=1)
    else:
        raise ValueError("mode는 'z' 또는 'radius'")

    # 클리핑(이상치 제거) -> 컬러 대비 좋아짐
    lo, hi = np.percentile(depth, clip_percent)
    depth = np.clip(depth, lo, hi)

    if invert:
        depth = -depth

    # 0~1 정규화
    dmin, dmax = float(depth.min()), float(depth.max())
    denom = (dmax - dmin) if (dmax - dmin) > 1e-12 else 1.0
    t = (depth - dmin) / denom  # 0..1

    # 간단 컬러맵 (blue->cyan->green->yellow->red)
    # t: 0(가까움/작음) ~ 1(멀음/큼)
    colors = np.zeros((t.shape[0], 3), dtype=np.float64)
    # piecewise
    # 0~0.25: blue->cyan
    m = t < 0.25
    u = (t[m] / 0.25)
    colors[m] = np.stack([0*u, u, 1*np.ones_like(u)], axis=1)

    # 0.25~0.5: cyan->green
    m = (t >= 0.25) & (t < 0.5)
    u = (t[m] - 0.25) / 0.25
    colors[m] = np.stack([0*u, 1*np.ones_like(u), (1-u)], axis=1)

    # 0.5~0.75: green->yellow
    m = (t >= 0.5) & (t < 0.75)
    u = (t[m] - 0.5) / 0.25
    colors[m] = np.stack([u, 1*np.ones_like(u), 0*u], axis=1)

    # 0.75~1.0: yellow->red
    m = t >= 0.75
    u = (t[m] - 0.75) / 0.25
    colors[m] = np.stack([1*np.ones_like(u), (1-u), 0*u], axis=1)

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return (dmin, dmax, lo, hi)

def visualize_depth_asc(asc_path,
                        ply_out_path=None,
                        xyz_cols=(0, 1, 2),
                        skiprows=0,
                        voxel_down=0.0,
                        mode="z",                 # "z" 또는 "radius"
                        clip_percent=(2, 98),
                        invert=False):
    pts = load_asc_points(asc_path, xyz_cols=xyz_cols, skiprows=skiprows)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    if voxel_down and voxel_down > 0:
        pcd = pcd.voxel_down_sample(float(voxel_down))

    dmin, dmax, lo, hi = apply_depth_colormap(
        pcd, mode=mode, clip_percent=clip_percent, invert=invert
    )

    print(f"[Depth mode] {mode}")
    print(f"[Depth raw min/max] {dmin:.4f} ~ {dmax:.4f} (after clip)")
    print(f"[Clip percentile] {clip_percent} -> {lo:.4f} ~ {hi:.4f}")
    print(f"[Points] {len(pcd.points)}")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"Depth View ({mode}) - {os.path.basename(asc_path)}"
    )

    if ply_out_path is None:
        base, _ = os.path.splitext(asc_path)
        ply_out_path = base + "_depth.ply"

    ok = o3d.io.write_point_cloud(ply_out_path, pcd, write_ascii=False)
    if not ok:
        raise IOError(f"PLY 저장 실패: {ply_out_path}")
    print(f"[OK] saved: {ply_out_path}")

if __name__ == "__main__":
    asc_path = r"C:\Users\SehoonKang\Desktop\test01.asc"
    visualize_depth_asc(
        asc_path,
        ply_out_path=r"C:\Users\SehoonKang\Desktop\becktron.ply",
        voxel_down=0.0,        # 너무 크면 0.5~2.0 정도로 올려봐
        mode="z",              # "z" 또는 "radius"
        clip_percent=(2, 98),  # 대비 좋게
        invert=False           # 깊이 방향 반대면 True
    )