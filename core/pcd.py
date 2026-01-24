from itertools import permutations
import json
import os
import re
from tqdm import tqdm
import imageio.v2 as iio
import numpy as np
import open3d as o3d
import cv2
import copy
from typing import Any, Dict, List, Optional, Tuple
from .util.circleCenter import fit_circle_center_3d
from .util.transform_utils import calculate_transformation_matrix
CircleResult = Tuple[int, int, float, float, float]

class PCD:
    def __init__(self):
        self._verbose = True
        self._verbose_transform = True
        return
    
    def set_path(self, body_path):
        self._body_path = body_path


    def merge_pcd(self, source_list : str, calibration_file : str, robotType : str, current_model : str) :
        path_dict = {}
        pcd_dict = {}
        T_base_cam_dict = {}
        detected_circle_centers = {}
        
        for frame in tqdm(source_list, total = len(source_list)) :
            texture_path, x_path, y_path, z_path, pose_path, mask_path = frame

            frame_number = os.path.basename(texture_path).replace("_IMG_Texture_8Bit.png", "")
            path_dict[frame_number] = (x_path, y_path, z_path, texture_path)
            detect_circle_setting = dict(topk_hough=5, roi_scale=2.2, roi_half_min=60,
                                         roi_half_max=260, band_px=4.0, arc_min=0.15,
                                         dp=1.2, minDist=140, param1=120,
                                         param2=24, minRadius=50, maxRadius=60)
            
            detected_circle_centers.setdefault(frame_number, [])
            
            if current_model == 'LH' :                
                if frame_number == '1' :
                    detect_circle_setting['minRadius'] = 10
                    circles2d = self.find_circle_center(texture_image=texture_path, detect_circle_setting=detect_circle_setting,x= 1527, y= 1169, r = 100)
                    if circles2d:
                        detected_circle_centers[frame_number].append(circles2d[0])
                elif frame_number == '3':
                    circles2d = self.find_circle_center(texture_image=texture_path, detect_circle_setting=detect_circle_setting, x= 2110, y=1010, r=100)
                    if circles2d:
                        detected_circle_centers[frame_number].append(circles2d[0])
                elif frame_number == '5':                
                    circles2d = self.find_circle_center(texture_image=texture_path, detect_circle_setting=detect_circle_setting, x=1166, y=844, r= 100)
                    if circles2d:
                        detected_circle_centers[frame_number].append(circles2d[0])
            else :
                if frame_number == '1':
                    detect_circle_setting = dict(
                            topk_hough=5,
                            roi_scale=2.6,          # 2.2 -> 2.6 (바깥 그림자까지 ROI에 더 여유)
                            roi_half_min=80,        # 60 -> 80 (너무 타이트하면 바깥원 잘림)
                            roi_half_max=300,       # 260 -> 300
                            band_px=6.0,            # 4.0 -> 6.0 (검증 밴드 넓혀서 바깥 에지 포함)
                            arc_min=0.10,           # 0.15 -> 0.10 (그림자 때문에 원호가 덜 잡혀도 통과)
                            dp=1.2,
                            minDist=120,            # 140 -> 120 (큰 의미 없지만 ROI내 1개면 낮춰도 됨)
                            param1=70,              # 120 -> 90 (에지 더 뽑히게)
                            param2=16,              # 24 -> 16 (누적 임계 낮춰 바깥 원 같은 약한 원도 후보로)
                            minRadius=20,           # ✅ 핵심: inner edge 배제
                            maxRadius=60          # ✅ 핵심: 바깥 원 범위로 제한
                        )
                    circles2d = self.find_circle_center(texture_image=texture_path, detect_circle_setting=detect_circle_setting, x=432, y = 1102, r = 100)
                    if circles2d:
                        detected_circle_centers[frame_number].append(circles2d[0])
                    circles2d = self.find_circle_center(texture_image=texture_path, detect_circle_setting=detect_circle_setting, x=642, y = 701, r = 100)
                    if circles2d:
                        detected_circle_centers[frame_number].append(circles2d[0])
                    circles2d = self.find_circle_center(texture_image=texture_path, detect_circle_setting=detect_circle_setting, x=806, y = 911, r = 100)
                    if circles2d:
                        detected_circle_centers[frame_number].append(circles2d[0])                    
                elif frame_number == '3':
                    circles2d = self.find_circle_center(texture_image=texture_path, detect_circle_setting=detect_circle_setting, x=2106, y = 673, r = 100)
                    if circles2d:
                        detected_circle_centers[frame_number].append(circles2d[0])
                elif frame_number == '5':
                    circles2d = self.find_circle_center(texture_image=texture_path, detect_circle_setting=detect_circle_setting, x = 1172, y = 802, r=100)
                    if circles2d:
                        detected_circle_centers[frame_number].append(circles2d[0])

            pose = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', open(pose_path, 'r', encoding='utf-8').read())]            
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

        merged_pcd, T_acc_list_master, T_acc_list_source = self._icp_merge(pcd_dict=pcd_dict)        
        
        def extract_index(group):            
            fname = os.path.basename(group[0])
            m = re.match(r"(\d+)_", fname)
            return int(m.group(1)) if m else -1
        
        source_list = sorted(source_list, key=extract_index)       
        
        T_list = T_acc_list_master + T_acc_list_source
        T_List_v2 = []
        for idx, source in enumerate(source_list):
            T_base_cam = T_base_cam_dict[str(idx+1)]
            T_acc = T_list[idx]['Transform']
            T_List_v2.append(T_acc @ T_base_cam)

        o3d.io.write_point_cloud(self._body_path, merged_pcd, print_progress=True)

        T_acc = T_acc_list_master + T_acc_list_source
        T_acc_sorted = sorted(T_acc, key=lambda d: int(d["number"]))
        T_cam_to_merged_dict = {}

        for d in T_acc_sorted:
            n = str(d["number"])            
            T_cam_to_merged_dict[n] = d["Transform"] @ T_base_cam_dict[n]
        circle_points_merged = []
        
        ANGLE_RANGE_BY_NAME = {
            "1": [(0, 360)],
            "2": [(0, 360)],
            "3": [(0, 360)], 
            "4": [(0, 360)],
            "5": [(0, 360)],
        }
        
        for frame_number in detected_circle_centers:
            if not detected_circle_centers[frame_number]:
                continue

            x_path, y_path, z_path, _ = path_dict[frame_number]
            X = iio.imread(x_path).astype(np.float32)
            Y = iio.imread(y_path).astype(np.float32)
            Z = iio.imread(z_path).astype(np.float32)

            for i  in range(len(detected_circle_centers[frame_number])) :
                gx, gy, rr, score, arc = detected_circle_centers[frame_number][i]

                p_cam, xyz = self.estimate_center_xyz_from_circle_ring_by_name(
                    X, Y, Z,
                    circle_cx=gx, circle_cy=gy, circle_r=rr,
                    name=frame_number,
                    angle_range_by_name=ANGLE_RANGE_BY_NAME,
                    n_samples=80,
                    win=10,
                    min_valid=15,
                    agg="median",
                    jitter_px=0.5
                )


                if p_cam is None:
                    print(f"[CIRCLE] invalid XYZ at ({gx},{gy}) in {frame_number}")
                    continue

                # self.show_ring_points_3d(xyz, p_cam)

                T_cam_to_merged = T_cam_to_merged_dict[frame_number]
                p_cam_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=np.float64)
                p_merged = (T_cam_to_merged @ p_cam_h)[:3]
                circle_points_merged.append(p_merged)

        T_list = [
            v for _, v in sorted(
                T_cam_to_merged_dict.items(),
                key=lambda kv: int(kv[0])
            )
        ]
        
        return T_List_v2, merged_pcd, circle_points_merged

    def _icp_merge(self, pcd_dict : dict[int, object]):
        master_frame_number = [1, 2, 3, 4, 5]

        master_merge_frame_list = [{"number" : str(n), "pcd" : pcd_dict[str(n)]} for n in master_frame_number]
        source_merge_frame_list = []

        for frame_number in sorted(pcd_dict.keys(), key=lambda f : int(f)) :
            if int(frame_number) in master_frame_number:
                continue
            source_merge_frame_list.append({"number" : frame_number, "pcd" : pcd_dict[frame_number]})
        merged_master_frames, T_acc_master_list = self._icp_merge_master_frames(master_frames=master_merge_frame_list)
        merged_all, T_acc_source_list = self._icp_merge_source_frames(merged_master=merged_master_frames, source_frames=source_merge_frame_list)
        print("align finish")
        return merged_all, T_acc_master_list, T_acc_source_list
    
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
    
    def crop_roi_center_px(self, img: np.ndarray, cx: int, cy: int, r_px: int) -> Tuple[np.ndarray, int, int]:
        """
        (cx,cy)를 중심으로 r_px 반경의 사각 ROI를 crop.
        return: (roi_img, x_off, y_off)  # roi의 좌상단 오프셋
        """
        h, w = img.shape[:2]
        x1 = max(cx - r_px, 0)
        y1 = max(cy - r_px, 0)
        x2 = min(cx + r_px + 1, w)
        y2 = min(cy + r_px + 1, h)
        roi = img[y1:y2, x1:x2].copy()
        return roi, x1, y1
    
    def find_circle_center(self, texture_image, detect_circle_setting, x, y, r) :        
        img = cv2.imread(texture_image, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {texture_image}")
        
        circles = self._get_circle_candidates(img, seed_x=x, seed_y=y, crop_r_px=r,
                                                  dp=detect_circle_setting['dp'],
                                                  minDist=detect_circle_setting["minDist"],
                                                  param1=detect_circle_setting["param1"], 
                                                  param2=detect_circle_setting["param2"], 
                                                  minRadius=detect_circle_setting["minRadius"], 
                                                  maxRadius=detect_circle_setting["maxRadius"],
        )

        if circles is None or len(circles) == 0:
            return []
        
        center_candidates = self._get_center_points(circles=circles, topk=detect_circle_setting["topk_hough"])
        if len(center_candidates) == 0:
            return []

        results: List[CircleResult] = []        

        for (cx, cy, r) in center_candidates:
            half = int(round(r * detect_circle_setting["roi_scale"]))
            half = max(detect_circle_setting["roi_half_min"], min(half, detect_circle_setting["roi_half_max"]))            
            vis_final = img.copy()       
            roi, x_off, y_off = self.crop_roi(img, cx, cy, half)
            if roi.size == 0:
                continue

            _, edge_img = self.draw_roi_contours(roi)

            cx_local = cx - x_off
            cy_local = cy - y_off

            best = self.pick_circle_by_annulus_fit(
                edge_img,
                seed_cx=cx_local,
                seed_cy=cy_local,
                seed_r=r,
                band_px=detect_circle_setting["band_px"],
                arc_min=detect_circle_setting["arc_min"]
            )

            if best is None:
                continue

            rx, ry, rr, score, arc_cov = best
            gx = rx + x_off
            gy = ry + y_off
            

            if self._verbose == True:
                cv2.circle(vis_final, (gx, gy), int(round(rr)), (0, 255, 0), 2)
                cv2.circle(vis_final, (gx, gy), 2, (0, 255, 0), -1)
                cv2.imshow("final(best overall)", vis_final)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            results.append((gx, gy, rr, score, arc_cov))

        results.sort(key=lambda t: t[3], reverse=True)  # score desc
        return results
    
    def _get_circle_candidates(self, img_bgr, seed_x: int, seed_y: int, crop_r_px: int,  dp=1.2, minDist=140, param1=120, param2=24, minRadius=50, maxRadius=60):
        roi, x_off, y_off = self.crop_roi_center_px(img_bgr, seed_x, seed_y, int(crop_r_px))
        if roi.size == 0:
            return []       
    
        # cv2.imshow("ROI (circle search area)", roi)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
        
        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius
        )

        if circles is None:
            return []

        circles = circles[0].astype(np.float32)
        circles[:, 0] += x_off
        circles[:, 1] += y_off
        return circles
        
    def _get_center_points(self, circles:np.ndarray, topk:int = 10):
        if circles is None:
            return []

        circles = np.squeeze(circles, axis=0) if circles.ndim == 3 else circles
        circles_sorted = sorted(
            circles,
            key=lambda c: c[2],
            reverse= True
        )[:topk]

        return [(int(round(x)), int(round(y)), int(round(r))) for x, y, r in circles_sorted]
    
    def crop_roi(self, img: np.ndarray, cx: int, cy: int, half: int) -> Tuple[np.ndarray, int, int]:
        h, w = img.shape[:2]
        x1 = max(cx - half, 0)
        y1 = max(cy - half, 0)
        x2 = min(cx + half + 1, w)
        y2 = min(cy + half + 1, h)
        roi = img[y1:y2, x1:x2].copy()
        return roi, x1, y1
    
    def draw_roi_contours(self, roi_bgr: np.ndarray):
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)

        edges = cv2.Canny(blur, 60, 180)
        
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        canvas = np.zeros_like(gray)
        cv2.drawContours(canvas, contours, -1, 255, thickness=1)

        return canvas, edges
    
    def pick_circle_by_annulus_fit(
        self,
        edge_img: np.ndarray,
        seed_cx: int,
        seed_cy: int,
        seed_r: int,
        *,
        band_px: float = 4.0,     # r0±band 안의 점만 사용 (에지 두께/노이즈에 따라 4~10)
        max_points: int = 4000,
        bins: int = 72,
        arc_min: float = 0.20,    # 반원도 살리려면 0.15~0.25
    ) -> Optional[Tuple[int, int, float, float, float]]:
        """
        edge_img에서 seed 중심/반지름 근처(annulus)의 점만 사용해 원 피팅.
        반환: (cx, cy, r, score, arc_cov)  # ROI 좌표계
        """
        ys, xs = np.where(edge_img > 0)
        if xs.size < 50:
            return None

        pts = np.stack([xs, ys], axis=1).astype(np.float64)

        # ✅ annulus 필터: seed 중심 기준 거리 r0±band만 남김
        dx = pts[:, 0] - float(seed_cx)
        dy = pts[:, 1] - float(seed_cy)
        dist = np.sqrt(dx*dx + dy*dy)

        rmin = max(3.0, seed_r - band_px)
        rmax = seed_r + band_px
        mask = (dist >= rmin) & (dist <= rmax)
        pts = pts[mask]
        if pts.shape[0] < 30:
            return None

        # 너무 많으면 샘플링
        if pts.shape[0] > max_points:
            idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
            pts = pts[idx]

        model = self.fit_circle_kasa(pts)
        if model is None:
            return None
        cx, cy, r = model

        # arc coverage 계산(피팅된 중심 기준)
        ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        idx = ((ang + np.pi) * (bins / (2 * np.pi))).astype(int)
        idx = np.clip(idx, 0, bins - 1)
        arc_cov = float(np.unique(idx).size / bins)
        if arc_cov < arc_min:
            return None

        # score: seed와 얼마나 일관적인지(중심 이동 + 반지름 차이) + arc
        center_shift = np.hypot(cx - seed_cx, cy - seed_cy)
        radius_err = abs(r - seed_r)
        # 간단 점수(낮을수록 좋은 에러들을 1/(1+err)로 변환)
        score = (1.0 / (1.0 + 0.6*center_shift + 0.8*radius_err)) * (0.7 + 0.3*arc_cov)

        return int(round(cx)), int(round(cy)), float(r), float(score), float(arc_cov)
    
    def fit_circle_kasa(self, points_xy: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """x^2+y^2 = a x + b y + c (Kasa)"""
        if points_xy.shape[0] < 20:
            return None
        x = points_xy[:, 0].astype(np.float64)
        y = points_xy[:, 1].astype(np.float64)
        A = np.c_[x, y, np.ones_like(x)]
        b = x*x + y*y
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        a, b2, c = sol
        cx = 0.5 * a
        cy = 0.5 * b2
        r2 = cx*cx + cy*cy + c
        if r2 <= 1e-9 or not np.isfinite(r2):
            return None
        return float(cx), float(cy), float(np.sqrt(r2))
    
    def estimate_center_xyz_from_circle_ring_by_name(self,
        X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
        circle_cx: float, circle_cy: float, circle_r: float,
        name: str,
        angle_range_by_name: dict,
        default_angle_ranges=[(0, 360)],
        n_samples: int = 80,
        win: int = 8,
        min_valid: int = 15,
        agg: str = "median",
        jitter_px: float = 0.8,
        rng: Optional[np.random.Generator] = None
    ) -> Optional[np.ndarray]:
        h, w = X.shape

        angle_ranges = angle_range_by_name.get(str(name), default_angle_ranges)

        pts_xy = self.sample_points_on_circle_pixels_with_angle_ranges(
            circle_cx, circle_cy, circle_r,
            n=n_samples, img_w=w, img_h=h,
            angle_ranges_deg=angle_ranges,
            jitter_px=jitter_px, rng=rng
        )

        xyz_list = []
        for (fx, fy) in pts_xy:
            px = int(round(fx))
            py = int(round(fy))
            p = self.pick_valid_xyz_from_pixel_mm(X, Y, Z, px, py, win=win)
            if p is not None:
                xyz_list.append(p)

        if len(xyz_list) < min_valid:
            return None

        xyz = np.stack(xyz_list, axis=0)
        # xyz = self.filter_ring_outliers_by_radial_mad(xyz, k=3.0, min_keep=min_valid)
        # center3d = self.fit_circle_center_3d_from_ring_points(xyz)

        center3d = fit_circle_center_3d(xyz_list)
        if center3d is None:
            return None, xyz
        return center3d, xyz
    
    def sample_points_on_circle_pixels_with_angle_ranges(self, 
        cx: float, cy: float, r: float,
        n: int,
        img_w: int, img_h: int,
        angle_ranges_deg: List[Tuple[float, float]],
        jitter_px: float = 0.8,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        지정한 각도 구간(도)에서만 원 둘레 픽셀 좌표 n개 샘플링.
        angle_ranges_deg: [(start_deg,end_deg), ...] (wrap 허용)
        """
        if rng is None:
            rng = np.random.default_rng()

        ranges_rad = self._deg_ranges_to_theta(angle_ranges_deg)

        # 각도 샘플을 n개 만들기: 구간별 길이(가중치)로 섞어서 뽑기
        # wrap(예: 300~60)은 [300~360] + [0~60]으로 분해
        expanded = []
        lengths = []
        for a, b in ranges_rad:
            if b >= a:
                expanded.append((a, b))
                lengths.append(b - a)
            else:
                # wrap
                expanded.append((a, 2*np.pi))
                lengths.append(2*np.pi - a)
                expanded.append((0.0, b))
                lengths.append(b - 0.0)

        lengths = np.array(lengths, dtype=np.float64)
        probs = lengths / lengths.sum()

        # 어떤 구간에서 뽑을지 n번 선택
        idxs = rng.choice(len(expanded), size=n, p=probs, replace=True)

        thetas = np.empty(n, dtype=np.float64)
        for i, k in enumerate(idxs):
            a, b = expanded[k]
            thetas[i] = rng.uniform(a, b)

        xs = cx + r * np.cos(thetas)
        ys = cy + r * np.sin(thetas)

        if jitter_px > 0:
            xs += rng.normal(0.0, jitter_px, size=n)
            ys += rng.normal(0.0, jitter_px, size=n)

        xs = np.clip(xs, 0, img_w - 1)
        ys = np.clip(ys, 0, img_h - 1)

        return np.stack([xs, ys], axis=1)

    def _deg_ranges_to_theta(self, ranges_deg: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """(deg_start, deg_end) 리스트를 (rad_start, rad_end)로. wrap은 그대로 두고 나중에 처리."""
        out = []
        for a, b in ranges_deg:
            out.append((np.deg2rad(a), np.deg2rad(b)))
        return out
    
    def pick_valid_xyz_from_pixel_mm(self,
        X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
        px: float, py: float, win: int = 8
    ) -> Optional[np.ndarray]:
        Z_MIN = 100.0
        Z_MAX = 3000.0
        XY_ABS_MAX = 6000.0

        h, w = X.shape
        px = int(round(px))
        py = int(round(py))
        x1 = max(0, px - win)
        x2 = min(w, px + win + 1)
        y1 = max(0, py - win)
        y2 = min(h, py + win + 1)

        Xw = X[y1:y2, x1:x2]
        Yw = Y[y1:y2, x1:x2]
        Zw = Z[y1:y2, x1:x2]

        valid = np.isfinite(Xw) & np.isfinite(Yw) & np.isfinite(Zw)
        valid &= ((Xw != 0) | (Yw != 0) | (Zw != 0))
        valid &= (Zw > Z_MIN) & (Zw < Z_MAX) & (np.abs(Xw) < XY_ABS_MAX) & (np.abs(Yw) < XY_ABS_MAX)

        if np.count_nonzero(valid) < 5:
            return None

        xs = Xw[valid].astype(np.float64)
        ys = Yw[valid].astype(np.float64)
        zs = Zw[valid].astype(np.float64)
        return np.array([np.median(xs), np.median(ys), np.median(zs)], dtype=np.float64)
    
    def filter_ring_outliers_by_radial_mad(self, 
        pts3d: np.ndarray,
        *,
        k: float = 3.0,
        min_keep: int = 12,
    ) -> np.ndarray:
        """
        링(원)에서 반지름이 튀는 점 제거.
        - 3D pts -> 평면 fit -> 2D 투영
        - 2D에서 LS 원(center,r) 대충 1번 맞춤
        - 각 점의 반지름 residual = |ri - median(ri)|
        - MAD 기반으로 outlier 제거

        return: filtered pts3d (K,3). 너무 적으면 원본 그대로 반환.
        """
        pts3d = np.asarray(pts3d, dtype=np.float64)
        if pts3d.ndim != 2 or pts3d.shape[1] != 3:
            raise ValueError("pts3d must be (N,3)")
        if pts3d.shape[0] < min_keep:
            return pts3d

        # 1) plane fit + basis
        c_plane, n = self.fit_plane_svd(pts3d)
        u, v = self.make_plane_basis(n)

        # 2) project to 2D
        P = pts3d - c_plane
        xy = np.column_stack([P @ u, P @ v])  # (N,2)

        # 3) rough circle fit in 2D
        out = self.fit_circle_2d_kasa(xy)
        if out is None:
            return pts3d
        c2, _r = out

        # 4) radius residuals
        ri = np.linalg.norm(xy - c2[None, :], axis=1)
        med = np.median(ri)
        abs_dev = np.abs(ri - med)

        mad = np.median(abs_dev) + 1e-12
        thr = k * 1.4826 * mad  # MAD -> sigma 근사

        keep = abs_dev <= thr
        filtered = pts3d[keep]

        # 너무 많이 날리면 오히려 망가질 수 있어서 안전장치
        if filtered.shape[0] < min_keep:
            return pts3d

        return filtered
    
    def fit_plane_svd(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        pts: (N,3)
        return: (c, n) where plane is (x-c)·n = 0
        """
        c = pts.mean(axis=0)
        P = pts - c
        _, _, Vt = np.linalg.svd(P, full_matrices=False)
        n = Vt[2, :]
        n = n / (np.linalg.norm(n) + 1e-12)
        return c, n

    def make_plane_basis(self, n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        n: unit normal
        return: (u, v) orthonormal basis on plane
        """
        # pick helper not parallel to n
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(helper, n)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = np.cross(n, helper)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(n, u)
        v = v / (np.linalg.norm(v) + 1e-12)
        return u, v

    def fit_circle_2d_kasa(self, xy: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """
        Kåsa least squares circle fit.
        xy: (N,2)
        return: (center_2d, radius) or None
        """
        if xy.shape[0] < 3:
            return None
        x = xy[:, 0]
        y = xy[:, 1]
        A = np.column_stack([2*x, 2*y, np.ones_like(x)])
        b = x**2 + y**2
        # Solve A p = b  (p = [cx, cy, c])
        p, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, c = p
        r2 = c + cx**2 + cy**2
        if r2 <= 0:
            return None
        r = float(np.sqrt(r2))
        return np.array([cx, cy], dtype=np.float64), r

    def fit_circle_center_3d_from_ring_points(self, pts3d: np.ndarray) -> Optional[np.ndarray]:
        """
        pts3d: (N,3) ring points on surface
        return: center3d (3,)
        """
        if pts3d.shape[0] < 10:
            return None

        # 1) fit plane
        c_plane, n = self.fit_plane_svd(pts3d)

        # 2) basis on plane
        u, v = self.make_plane_basis(n)

        # 3) project to plane coords (2D)
        P = pts3d - c_plane
        xy = np.column_stack([P @ u, P @ v])  # (N,2)

        # 4) fit circle in 2D
        out = self.fit_circle_2d_kasa(xy)
        if out is None:
            return None
        center2d, _r = out

        # 5) back to 3D
        center3d = c_plane + center2d[0]*u + center2d[1]*v
        return center3d

    def show_ring_points_3d(self,                            
        ring_pts3d: np.ndarray,
        center3d: Optional[np.ndarray] = None,
        *,
        ring_color=(0.2, 1.0, 0.2),
        center_color=(1.0, 0.2, 0.2),
        frame_size=50.0,
        center_sphere_radius=5.0,
    ):
        """
        estimate_center_xyz_from_circle_ring_by_name(..., return_ring=True)로 받은
        ring_pts3d를 그대로 3D로 보여준다.
        """
        print("----------------------------------")

        if ring_pts3d is None or len(ring_pts3d) == 0:
            print("[SHOW] ring_pts3d is empty.")
            return

        ring_pts3d = np.asarray(ring_pts3d, dtype=np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ring_pts3d)
        pcd.paint_uniform_color(list(ring_color))

        geoms = [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(frame_size))]

        if center3d is not None:
            center3d = np.asarray(center3d, dtype=np.float64).reshape(3)
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=float(center_sphere_radius))
            sph.translate(center3d)
            sph.paint_uniform_color(list(center_color))
            geoms.append(sph)

        o3d.visualization.draw_geometries(geoms)

    def make_points_pcd(self, points_xyz: np.ndarray, color=(1.0, 0.0, 0.0)) -> o3d.geometry.PointCloud:        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
        cols = np.tile(np.array(color, dtype=np.float64).reshape(1, 3), (len(points_xyz), 1))
        pcd.colors = o3d.utility.Vector3dVector(cols)
        return pcd
    
    def move_merged_pcd_to_cad(
        self,
        merged_pcd: o3d.geometry.PointCloud,
        CAD_CENTERS: np.ndarray,     # (3,3) CAD 기준 3점
        align_points: np.ndarray,    # (3,3) merged 좌표계에서 얻은 3점 (순서 섞여도 OK)
        *,
        copy_pcd: bool = True,       # True면 원본 보존
        return_report: bool = True
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray, Optional[Dict[str, Any]]]:
        if merged_pcd is None:
            raise ValueError("merged_pcd가 None 입니다.")
        if len(merged_pcd.points) == 0:
            raise ValueError("merged_pcd에 포인트가 없습니다.")

        # dummy records (함수 시그니처 맞추기용) - 실제로는 PCD 변환만 필요
        dummy_records = [
            {"index": i, "center": align_points[i].tolist()}
            for i in range(align_points.shape[0])
        ]

        # ✅ measured(=merged) -> CAD 변환 T 구하기
        T_meas_to_cad, _, report = self.align_centers_by_3points_auto_permute(
            CAD_CENTERS=CAD_CENTERS,
            align_points=align_points,
            measured_center_records=dummy_records
        )

        T_meas_to_cad = np.asarray(T_meas_to_cad, dtype=np.float64)

        # ✅ merged_pcd 전체에 변환 적용
        moved_pcd = o3d.geometry.PointCloud(merged_pcd) if copy_pcd else merged_pcd
        moved_pcd.transform(T_meas_to_cad)

        if not return_report:
            report = None

        return moved_pcd, T_meas_to_cad, report
    
    def move_merged_pcd_to_cad_v2(
        self,
        merged_pcd: o3d.geometry.PointCloud,
        master_cad_scale,
        CAD_CENTERS: np.ndarray,     # (3,3) CAD 기준 3점
        align_points: np.ndarray,    # (3,3) merged 좌표계에서 얻은 3점 (순서 섞여도 OK)
        *,
        copy_pcd: bool = True,       # True면 원본 보존
        return_report: bool = False
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray, Optional[Dict[str, Any]]]:
        if merged_pcd is None:
            raise ValueError("merged_pcd가 None 입니다.")
        if len(merged_pcd.points) == 0:
            raise ValueError("merged_pcd에 포인트가 없습니다.")
        
        # if self._verbose_transform:
        #     T_meas_to_cad, report = compute_transformation_matrix_with_verification(source_points=align_points, master_points=CAD_CENTERS)
        # else:
        #     T_meas_to_cad = compute_transformation_matrix(source_points=align_points, master_points=CAD_CENTERS)
        

        # 점의 순서가 맞지 않아 source_order, master_order 를 생성함
        transformation_matrix, aligned_points, errors = calculate_transformation_matrix(
            source_points_original= align_points, 
            master_points_original = CAD_CENTERS,
            source_order=[2, 1, 0],
            master_order=[0, 2, 1],
            # master_scale=1.0055786,
            master_scale=master_cad_scale,
            verbose=self._verbose
        )
        T_meas_to_cad = transformation_matrix
        


        # # dummy records (함수 시그니처 맞추기용) - 실제로는 PCD 변환만 필요
        # dummy_records = [
        #     {"index": i, "center": align_points[i].tolist()}
        #     for i in range(align_points.shape[0])
        # ]

        # # ✅ measured(=merged) -> CAD 변환 T 구하기
        # T_meas_to_cad, _, report = self.align_centers_by_3points_auto_permute(
        #     CAD_CENTERS=CAD_CENTERS,
        #     align_points=align_points,
        #     measured_center_records=dummy_records
        # )

        T_meas_to_cad = np.asarray(T_meas_to_cad, dtype=np.float64)

        # ✅ merged_pcd 전체에 변환 적용
        moved_pcd = o3d.geometry.PointCloud(merged_pcd) if copy_pcd else merged_pcd
        moved_pcd.transform(T_meas_to_cad)

        if not return_report:
            report = None

        return moved_pcd, T_meas_to_cad, report
    
    def align_centers_by_3points_auto_permute(self, # 이거는 detect -> cad로 변환하는 기존 구조.... 귀찮아서 위 함수(align_points_by_3ref_auto_permute) 통합 X
        CAD_CENTERS: np.ndarray,          # (3,3)
        align_points: np.ndarray,         # (3,3)
        measured_center_records: list     # [{"index": i, "center": [x,y,z]}, ...]
    ) -> Tuple[np.ndarray, list, Dict[str, Any]]:
        """
        반환:
        - T (4x4): measured -> CAD
        - moved_center_records: [{"index": i, "center": [x,y,z]}, ...]
        - report
        """
        CAD_CENTERS = np.asarray(CAD_CENTERS, dtype=np.float64)
        align_points = np.asarray(align_points, dtype=np.float64)

        if CAD_CENTERS.shape != (3, 3):
            raise ValueError("CAD_CENTERS는 (3,3) 이어야 합니다.")
        if align_points.shape != (3, 3):
            raise ValueError("align_points는 (3,3) 이어야 합니다.")
        if len(measured_center_records) == 0:
            raise ValueError("measured_center_records가 비어 있습니다.")

        # ---- measured centers 분리 ----
        meas_xyz = np.array(
            [r["center"] for r in measured_center_records],
            dtype=np.float64
        )
        meas_idx = [int(r["index"]) for r in measured_center_records]

        best = None  # (rmse, perm, T, aligned3)

        for perm in permutations([0, 1, 2], 3):
            src = align_points[list(perm), :]
            T = self.rigid_transform_kabsch(src_pts=src, dst_pts=CAD_CENTERS)
            aligned3 = self.apply_T(src, T)
            e = self._rmse(aligned3, CAD_CENTERS)

            if (best is None) or (e < best[0]):
                best = (e, perm, T, aligned3)

        best_rmse, best_perm, best_T, best_aligned3 = best

        # ---- 전체 measured center 이동 ----
        moved_xyz = self.apply_T(meas_xyz, best_T)

        moved_center_records = []
        for idx, xyz in zip(meas_idx, moved_xyz):
            moved_center_records.append({
                "index": idx,
                "center": xyz.tolist()
            })

        per_point_err = np.linalg.norm(best_aligned3 - CAD_CENTERS, axis=1)

        report = {
            "best_perm_src_index_order": list(best_perm),
            "rmse": float(best_rmse),
            "per_point_error": per_point_err.tolist(),
            "max_error": float(np.max(per_point_err)),
            "T_meas_to_cad": best_T.tolist(),
            "aligned3_after_T": best_aligned3.tolist(),
        }

        return best_T, moved_center_records, report

    def rigid_transform_kabsch(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        """
        src_pts (N,3) -> dst_pts (N,3) 로 보내는 4x4 rigid transform T를 구함.
        dst ≈ (T @ [src,1])[:3]
        """
        src_pts = np.asarray(src_pts, dtype=np.float64)
        dst_pts = np.asarray(dst_pts, dtype=np.float64)

        if src_pts.shape != dst_pts.shape or src_pts.shape[1] != 3:
            raise ValueError("src_pts와 dst_pts는 같은 shape (N,3) 이어야 합니다.")

        c_src = src_pts.mean(axis=0)
        c_dst = dst_pts.mean(axis=0)

        X = src_pts - c_src
        Y = dst_pts - c_dst

        H = X.T @ Y
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # reflection 방지
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = c_dst - R @ c_src

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def apply_T(self, points: np.ndarray, T: np.ndarray) -> np.ndarray:
        """(N,3) points에 4x4 T 적용해서 (N,3) 반환"""
        points = np.asarray(points, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)

        if points.ndim == 1:
            points = points.reshape(1, 3)

        ones = np.ones((points.shape[0], 1), dtype=np.float64)
        ph = np.hstack([points, ones])          # (N,4)
        out = (T @ ph.T).T[:, :3]               # (N,3)
        return out

    def _rmse(self, a: np.ndarray, b: np.ndarray) -> float:
        d = a - b
        per = np.linalg.norm(d, axis=1)
        return float(np.sqrt(np.mean(per ** 2)))
    
    def denoise_by_dbscan_keep_big_clusters(self,
        pcd: o3d.geometry.PointCloud,
        eps: float = 8.0,
        min_points: int = 30,
        keep_top_k: int = 1,
        min_cluster_size: int | None = None,
    ):
        """
        eps: 같은 군집으로 볼 거리(단위는 pcd 좌표 단위: 보통 mm)
        min_points: 군집 코어 포인트 기준
        keep_top_k: 가장 큰 군집 k개만 유지
        min_cluster_size: 이 값보다 작은 군집은 제거(keep_top_k 대신/추가로 사용 가능)

        return: (filtered_pcd, labels)
        """
        if len(pcd.points) == 0:
            return pcd, np.array([], dtype=int)

        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        # labels: -1 = noise

        valid = labels >= 0
        if not np.any(valid):
            # 전부 노이즈로 떨어짐 → 파라미터가 너무 타이트
            return pcd.select_by_index([], invert=False), labels

        # 각 클러스터 크기 계산
        max_label = labels.max()
        counts = np.bincount(labels[valid], minlength=max_label + 1)

        # 유지할 클러스터 선택
        keep = np.zeros_like(labels, dtype=bool)

        if min_cluster_size is not None:
            good_ids = np.where(counts >= int(min_cluster_size))[0]
            for cid in good_ids:
                keep |= (labels == cid)
        else:
            # 큰 순서대로 top-k
            top_ids = np.argsort(counts)[::-1][:int(keep_top_k)]
            for cid in top_ids:
                keep |= (labels == cid)

        idx_keep = np.where(keep)[0].tolist()
        filtered = pcd.select_by_index(idx_keep)

        return filtered, labels