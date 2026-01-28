import open3d as o3d

def load_ply(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"PointCloud is empty: {path}")
    return pcd

def visualize_two_plys(ply_a: str, ply_b: str, paint=True):
    pcd_a = load_ply(ply_a)
    pcd_b = load_ply(ply_b)

    # 보기 쉽게 색 입히기(각각 다른 색)
    # if paint:
    #     if not pcd_a.has_colors():
    # pcd_a.paint_uniform_color([0.75, 0.75, 0.75])  # red-ish
        # if not pcd_b.has_colors():
        #     pcd_b.paint_uniform_color([0.2, 0.6, 1.0])  # blue-ish

    # 좌표축
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0, origin=[0, 0, 0])

    o3d.visualization.draw_geometries(
        [pcd_a, pcd_b, axis],
        window_name="Two PLYs",
        point_show_normal=False
    )

if __name__ == "__main__":
    ply1 = r"C:\Users\SehoonKang\Desktop\becktron.ply"
    ply2 = r"C:\Users\SehoonKang\Desktop\real.ply"
    visualize_two_plys(ply1, ply2, paint=True)