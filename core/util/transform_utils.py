import numpy as np
from scipy.optimize import minimize, minimize_scalar

def calculate_transformation_matrix(source_points_original, master_points_original, 
                                    source_order=[2, 1, 0], master_order=[0, 2, 1],
                                    master_scale=1.0, verbose=False):
    """
    Source 점들을 Master 점들에 정렬하는 4x4 변환 행렬을 계산
    
    Parameters:
    -----------
    source_points_original : np.ndarray (3, 3)
        원본 source 3개 점 좌표
    master_points_original : np.ndarray (3, 3)
        원본 master 3개 점 좌표
    source_order : list
        source 점들의 순서 재배열 인덱스 (기본값: [2, 1, 0])
    master_order : list
        master 점들의 순서 재배열 인덱스 (기본값: [0, 2, 1])
    master_scale : float
        master 점들에 적용할 스케일 (기본값: 1.0)
    verbose : bool
        중간 결과 출력 여부 (기본값: False)
    
    Returns:
    --------
    transformation_matrix : np.ndarray (4, 4)
        4x4 동차 변환 행렬
    aligned_points : np.ndarray (3, 3)
        변환된 source 점들
    errors : dict
        각 점의 오차 정보
    """
    
    # 순서 조정
    source_points = np.array([source_points_original[i] for i in source_order])
    master_points = np.array([master_points_original[i] for i in master_order])
    master_points = master_points * master_scale
    
    def rodrigues_rotation(axis, angle):
        """Rodrigues' rotation formula로 회전 행렬 생성"""
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-10:
            return np.eye(3)
        axis = axis / axis_norm
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R
    
    # Step 1-2: 0번, 1번 점 균등 오차 정렬
    def objective_alignment(params):
        tx, ty, tz, rx, ry, rz, angle = params
        
        # 평행이동
        translation = np.array([tx, ty, tz])
        transformed = source_points + translation
        
        # 회전 (0번 점을 중심으로)
        if np.linalg.norm([rx, ry, rz]) > 1e-6:
            rotation_axis = np.array([rx, ry, rz])
            R = rodrigues_rotation(rotation_axis, angle)
            
            origin = transformed[0]
            transformed_centered = transformed - origin
            transformed_centered = (R @ transformed_centered.T).T
            transformed = transformed_centered + origin
        
        # 0번과 1번 점의 오차 계산
        error_0 = np.linalg.norm(transformed[0] - master_points[0])
        error_1 = np.linalg.norm(transformed[1] - master_points[1])
        
        # 목적함수: 전체 오차 + 오차 불균형 페널티
        total_error = error_0**2 + error_1**2
        balance_penalty = (error_0 - error_1)**2 * 0.5
        
        return total_error + balance_penalty
    
    # 초기값 설정
    initial_translation = master_points[0] - source_points[0]
    initial_params = [
        initial_translation[0], initial_translation[1], initial_translation[2],
        0.0, 0.0, 1.0,  # 초기 회전축
        0.0  # 초기 회전각
    ]
    
    # 최적화
    result = minimize(objective_alignment, initial_params, method='Powell', 
                      options={'maxiter': 5000, 'disp': False})
    
    optimal_params = result.x
    tx, ty, tz, rx, ry, rz, angle = optimal_params
    
    # 첫 번째 변환 적용
    translation_1 = np.array([tx, ty, tz])
    source_aligned = source_points + translation_1
    
    rotation_matrix_1 = np.eye(3)
    rotation_center_1 = source_aligned[0].copy()
    
    if np.linalg.norm([rx, ry, rz]) > 1e-6:
        rotation_axis = np.array([rx, ry, rz])
        rotation_matrix_1 = rodrigues_rotation(rotation_axis, angle)
        
        source_centered = source_aligned - rotation_center_1
        source_centered = (rotation_matrix_1 @ source_centered.T).T
        source_aligned = source_centered + rotation_center_1
    
    if verbose:
        error_0 = np.linalg.norm(source_aligned[0] - master_points[0])
        error_1 = np.linalg.norm(source_aligned[1] - master_points[1])
        print(f"Step 1-2: Point 0 error: {error_0:.2f}, Point 1 error: {error_1:.2f}")
        print(f"Error ratio: {error_0/error_1:.3f}")
    
    # Step 3: 0-1번 점을 축으로 2번 점 회전 최적화
    axis = source_aligned[1] - source_aligned[0]
    axis = axis / np.linalg.norm(axis)
    rotation_center_2 = source_aligned[0].copy()
    
    def rotate_around_axis(point, origin, axis, angle):
        point_centered = point - origin
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return (R @ point_centered) + origin
    
    def objective_point2(angle):
        rotated_point2 = rotate_around_axis(source_aligned[2], rotation_center_2, axis, angle)
        distance = np.linalg.norm(rotated_point2 - master_points[2])
        return distance
    
    result_point2 = minimize_scalar(objective_point2, bounds=(0, 2*np.pi), method='bounded')
    optimal_angle_point2 = result_point2.x
    
    # 두 번째 회전 행렬
    rotation_matrix_2 = rodrigues_rotation(axis, optimal_angle_point2)
    
    # 최종 변환 적용
    source_final = source_aligned.copy()
    source_final[2] = rotate_around_axis(source_aligned[2], rotation_center_2, axis, optimal_angle_point2)
    
    # 4x4 변환 행렬 생성
    # T = T3 * R2 * T2 * R1 * T1
    # T1: 평행이동 1
    T1 = np.eye(4)
    T1[:3, 3] = translation_1
    
    # R1: 회전 1 (rotation_center_1 기준)
    T_to_origin_1 = np.eye(4)
    T_to_origin_1[:3, 3] = -rotation_center_1
    
    R1 = np.eye(4)
    R1[:3, :3] = rotation_matrix_1
    
    T_from_origin_1 = np.eye(4)
    T_from_origin_1[:3, 3] = rotation_center_1
    
    # R2: 회전 2 (rotation_center_2 기준)
    T_to_origin_2 = np.eye(4)
    T_to_origin_2[:3, 3] = -rotation_center_2
    
    R2 = np.eye(4)
    R2[:3, :3] = rotation_matrix_2
    
    T_from_origin_2 = np.eye(4)
    T_from_origin_2[:3, 3] = rotation_center_2
    
    # 전체 변환 행렬 조합
    transformation_matrix = T_from_origin_2 @ R2 @ T_to_origin_2 @ T_from_origin_1 @ R1 @ T_to_origin_1 @ T1
    
    # 오차 계산
    errors = {}
    total_error = 0
    for i in range(3):
        dist = np.linalg.norm(source_final[i] - master_points[i])
        errors[f'point_{i}'] = dist
        total_error += dist**2
    errors['rms'] = np.sqrt(total_error / 3)
    
    if verbose:
        print("\n=== 최종 정렬 결과 ===")
        for i in range(3):
            print(f"Point {i} error: {errors[f'point_{i}']:.2f}")
        print(f"Total RMS error: {errors['rms']:.2f}")
    
    return transformation_matrix, source_final, errors


# 테스트 코드
if __name__ == "__main__":
    import open3d as o3d
    
    # 원본 포인트
    source_points_original = np.array([
        [527.62, 1069.1, -177.54],
        [848.68, 1073.1, -200.42],
        [128.81, 1102.5, -228.63]
    ])
    
    master_points_original = np.array([
        [512.69, -452.61, 1239.6],
        [912.63, -446.2, 1207.8],
        [1225.5, -379.29, 1214.1]
    ])
    
    # 변환 행렬 계산
    transformation_matrix, aligned_points, errors = calculate_transformation_matrix(
        source_points_original, 
        master_points_original,
        source_order=[2, 1, 0],
        master_order=[0, 2, 1],
        master_scale=1.0055786,
        verbose=True
    )
    
    print("\n=== 4x4 변환 행렬 ===")
    print(transformation_matrix)
    
    # 검증: 변환 행렬을 source_points_original에 적용
    source_reordered = np.array([
        source_points_original[2],
        source_points_original[1],
        source_points_original[0]
    ])
    
    # 동차 좌표로 변환
    source_homogeneous = np.hstack([source_reordered, np.ones((3, 1))])
    
    # 변환 적용
    transformed_homogeneous = (transformation_matrix @ source_homogeneous.T).T
    transformed_points = transformed_homogeneous[:, :3]
    
    print("\n=== 변환 행렬 검증 ===")
    for i in range(3):
        print(f"Transformed point {i}: {transformed_points[i]}")
        print(f"Aligned point {i}: {aligned_points[i]}")
        print(f"Difference: {np.linalg.norm(transformed_points[i] - aligned_points[i]):.6f}\n")
    
    # 시각화
    master_points = np.array([
        master_points_original[0],
        master_points_original[2],
        master_points_original[1]
    ]) * 1.0055786
    
    geometries = []
    
    # Source 원본 (초록색)
    source_original_pcd = o3d.geometry.PointCloud()
    source_original_pcd.points = o3d.utility.Vector3dVector(source_reordered)
    source_original_pcd.paint_uniform_color([0, 1, 0])
    
    # Source 변환 후 (빨간색)
    source_transformed_pcd = o3d.geometry.PointCloud()
    source_transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    source_transformed_pcd.paint_uniform_color([1, 0, 0])
    
    # Master (파란색)
    master_pcd = o3d.geometry.PointCloud()
    master_pcd.points = o3d.utility.Vector3dVector(master_points)
    master_pcd.paint_uniform_color([0, 0, 1])
    
    geometries.extend([source_original_pcd, source_transformed_pcd, master_pcd])
    
    # 구 추가
    for i, point in enumerate(source_reordered):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=20)
        sphere.translate(point)
        sphere.paint_uniform_color([0.5, 1, 0.5])
        geometries.append(sphere)
    
    for i, point in enumerate(transformed_points):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=20)
        sphere.translate(point)
        sphere.paint_uniform_color([1, 0.5, 0.5])
        geometries.append(sphere)
    
    for i, point in enumerate(master_points):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=20)
        sphere.translate(point)
        sphere.paint_uniform_color([0.5, 0.5, 1])
        geometries.append(sphere)
    
    # 오차 선 (흰색)
    for i in range(3):
        error_line = o3d.geometry.LineSet()
        error_line.points = o3d.utility.Vector3dVector([transformed_points[i], master_points[i]])
        error_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        error_line.colors = o3d.utility.Vector3dVector([[1, 1, 1]])
        geometries.append(error_line)
    
    # 선 추가
    lines = [[0, 1], [1, 2], [2, 0]]
    
    # Source 원본
    source_line_set = o3d.geometry.LineSet()
    source_line_set.points = o3d.utility.Vector3dVector(source_reordered)
    source_line_set.lines = o3d.utility.Vector2iVector(lines)
    source_line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in lines])
    
    # Source 변환 후
    transformed_line_set = o3d.geometry.LineSet()
    transformed_line_set.points = o3d.utility.Vector3dVector(transformed_points)
    transformed_line_set.lines = o3d.utility.Vector2iVector(lines)
    transformed_line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])
    
    # Master
    master_line_set = o3d.geometry.LineSet()
    master_line_set.points = o3d.utility.Vector3dVector(master_points)
    master_line_set.lines = o3d.utility.Vector2iVector(lines)
    master_line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in lines])
    
    geometries.extend([source_line_set, transformed_line_set, master_line_set])
    
    # 좌표축
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
    geometries.append(coord_frame)
    
    print("\n초록색: Source 원본")
    print("빨간색: Source 변환 후")
    print("파란색: Master")
    print("흰색: 오차 표시 선")
    
    o3d.visualization.draw_geometries(geometries,
                                       window_name="Transformation Matrix Result",
                                       width=1200,
                                       height=800)