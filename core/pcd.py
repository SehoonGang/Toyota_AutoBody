import os
import re
from tqdm import tqdm
import imageio.v2 as iio
import numpy as np
import open3d as o3d
import cv2
import copy

class PCD:
    def __init__(self):
        return

    def merge_pcd(self, source_list : str, calibration_file : str, robotType : str) :
        path_dict = {}
        pcd_dict = {}
        T_base_cam_dict = {}
        
        for frame in tqdm(source_list, total = len(source_list)) :
            texture_path, x_path, y_path, z_path, pose_path, mask_path = frame

            frame_number = os.path.basename(texture_path).replace("_IMG_Texture_8Bit.png", "")
            path_dict[frame_number] = (x_path, y_path, z_path, texture_path)

            pose = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', open(pose_path, 'r', encoding='utf-8').read())]
            if len(pose) != 6:
                print("[POSE WARN]", frame_number, "len=", len(pose), pose[:10], "path=", pose_path)
            pose = pose[:6]
            pcd_cam = self._make_cam_pcd(x_path, y_path, z_path, texture_path, mask_path)

            robotType = robotType.lower()
            if robotType == "fanuc" :
                T_base_tcp = self._transform_fanuc_coords_to_T(*pose)

            T_tcp_cam = self._transform_calibration_file_to_T_4x4(calibration_file)
            T_base_cam = T_base_tcp @ T_tcp_cam
            pcd_base = copy.deepcopy(pcd_cam)
            pcd_base.transform(T_base_cam)
            pcd_dict[frame_number] = pcd_base
            T_base_cam_dict[frame_number] = T_base_cam 

        merged_pcd, _ = self._icp_merge(pcd_dict=pcd_dict)
        return merged_pcd
        

    def _icp_merge(self, pcd_dict : dict[int, object]):
        master_frame_number = [5, 4, 1, 2, 3, 9]

        master_merge_frame_list = [{"number" : str(n), "pcd" : pcd_dict[str(n)]} for n in master_frame_number]
        source_merge_frame_list = []

        for frame_number in sorted(pcd_dict.keys(), key=lambda f : int(f)) :
            if int(frame_number) in master_frame_number:
                continue
            source_merge_frame_list.append({"number" : frame_number, "pcd" : pcd_dict[frame_number]})
        merged_master_frames, T_acc_master_list = self._icp_merge_master_frames(master_frames=master_merge_frame_list)
        merged_all, T_acc_list_all = self._icp_merge_source_frames(merged_master=merged_master_frames, source_frames=source_merge_frame_list)

        return merged_all, T_acc_list_all
    
    def _icp_merge_master_frames(self, master_frames):
        merged_pcd = copy.deepcopy(master_frames[0]["pcd"])
        T_list = [{"number": master_frames[0]["number"], "Transform": np.eye(4)}]

        for i in range(1, len(master_frames)):
            source = master_frames[i]["pcd"]
            target = merged_pcd

            _, T_rel = self._icp_multistage_varying_voxel(
                source=source,
                target=target,
                init_T=np.eye(4)   # base에서 이미 맞았으니 I 근처 미세조정
            )

            src_aligned = copy.deepcopy(source)
            src_aligned.transform(T_rel)
            merged_pcd += src_aligned

            T_list.append({"number": master_frames[i]["number"], "Transform": T_rel})

        return merged_pcd, T_list
    
    def _icp_merge_source_frames(self, merged_master, source_frames):
        merged_pcd = copy.deepcopy(merged_master)
        T_list = []

        for sf in source_frames:
            source = sf["pcd"]
            target = merged_pcd  # 또는 merged_master (더 안정적/더 빠름은 merged_master)

            _, T_rel = self._icp_multistage_varying_voxel(
                source=source,
                target=target,
                init_T=np.eye(4)
            )

            src_aligned = copy.deepcopy(source)
            src_aligned.transform(T_rel)
            merged_pcd += src_aligned

            T_list.append({"number": sf["number"], "Transform": T_rel})

        return merged_pcd, T_list      

    def _icp_multistage_varying_voxel(self,
        source, target, init_T=None,
        stages=((2.0, 4.0, 20), (1.0, 2.0, 15), (0.5, 1.0, 10)),
        use_point_to_plane=True,
        normal_radius_mul=3.0,
        normal_max_nn=30,
    ):
        if init_T is None:
            init_T = np.eye(4, dtype=np.float64)

        T = init_T.copy()
        last = None

        for stage_idx, (voxel_mm, max_corr_mm, max_iter) in enumerate(stages, start=1):
            src = source.voxel_down_sample(float(voxel_mm))
            tgt = target.voxel_down_sample(float(voxel_mm))

            if use_point_to_plane:
                normal_radius_mm = float(voxel_mm * normal_radius_mul)
                if not src.has_normals():
                    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius_mm, max_nn=normal_max_nn))
                    src.normalize_normals()
                if not tgt.has_normals():
                    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius_mm, max_nn=normal_max_nn))
                    tgt.normalize_normals()
                estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
            else:
                estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iter))

            last = o3d.pipelines.registration.registration_icp(
                src, tgt,
                max_correspondence_distance=float(max_corr_mm),
                init=T,
                estimation_method=estimation,
                criteria=criteria
            )
            T = np.asarray(last.transformation, dtype=np.float64)

            corr = len(last.correspondence_set)
            print(f"[stage {stage_idx}] voxel={voxel_mm}mm corr={max_corr_mm}mm iter={max_iter} "
                f"corrN={corr} fitness={last.fitness:.6f} rmse={last.inlier_rmse:.6f}")

            if corr == 0:
                break

        return last, T

    def _transform_calibration_file_to_T_4x4(self, calibration_file : str, to_meters : bool = False):
        path = calibration_file
        key = "ArmTipToMarkerTagTransform"

        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

        if not fs.isOpened():
            raise IOError(f"열 수 없습니다: {path}")
        node = fs.getNode(key)
        if node.empty():
            fs.release()
            raise KeyError(f"키가 없습니다: {key}")
        T = node.mat().astype(np.float64)
        fs.release()

        if T.shape != (4, 4):
            raise ValueError(f"4x4 행렬이 아님: {T.shape}")

        if to_meters:
            T = T.copy()
            T[:3, 3] *= 1e-3

        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4) or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9):
            raise ValueError("Not a 4x4 homogeneous.")
        return T
    
    def _transform_fanuc_coords_to_T(self, x, y, z, w, p, r, degrees = True):
            W, P, R = (np.deg2rad([w, p, r]) if degrees else (w, p, r))
            cW, sW = np.cos(W), np.sin(W)
            cP, sP = np.cos(P), np.sin(P)
            cR, sR = np.cos(R), np.sin(R)

            Rx = np.array([[1, 0, 0],
                        [0, cW, -sW],
                        [0, sW,  cW]], dtype=np.float64)
            Ry = np.array([[ cP, 0, sP],
                        [  0, 1,  0],
                        [-sP, 0, cP]], dtype=np.float64)
            Rz = np.array([[cR, -sR, 0],
                        [sR,  cR, 0],
                        [0,    0, 1]], dtype=np.float64)

            Rm = Rz @ Ry @ Rx

            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rm
            T[:3, 3] = [x, y, z]
            return T

    def _make_cam_pcd(self, x_path, y_path, z_path, texture_path, mask_path, mask_zero_pad_px = 50):
        Z_MIN = 100.0
        Z_MAX = 3000.0
        XY_ABS_MAX = 6000.0    

        X = iio.imread(x_path).astype(np.float32)
        Y = iio.imread(y_path).astype(np.float32)
        Z = iio.imread(z_path).astype(np.float32)
        RGB = iio.imread(texture_path)[..., :3] / 255.0

        if mask_path and os.path.isfile(mask_path):
            M = iio.imread(mask_path)
            if M.ndim == 3:
                M = M[..., 0]
            mask = (M > 0)
        else:
            mask = np.ones_like(X, dtype=bool)

        if mask_zero_pad_px > 0:
            r = int(mask_zero_pad_px)
            k = 2 * r + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask_u8 = mask.astype(np.uint8)
            mask_u8 = cv2.erode(mask_u8, kernel, iterations=1)
            mask = mask_u8.astype(bool)

        pts = np.stack([X, Y, Z], axis=-1)
        valid = np.isfinite(pts).all(axis=2) & (np.linalg.norm(pts, axis=2) > 0) & mask

        pts = pts[valid]
        cols = RGB[valid]

        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        range_mask = ((z > Z_MIN) & (z < Z_MAX) & (np.abs(x) < XY_ABS_MAX) & (np.abs(y) < XY_ABS_MAX))
        pts = pts[range_mask]
        cols = cols[range_mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
        return pcd