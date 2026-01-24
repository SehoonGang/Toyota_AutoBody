import datetime
import json
import math
import os
import sys
from typing import Optional, Sequence, Union
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (    
    QApplication, QComboBox, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox,
    QPushButton, QRadioButton, QTextEdit,
    QVBoxLayout, QWidget, QButtonGroup,
    QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox
)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
import pyqtgraph.opengl as gl
import torch
from core import Utils, FileType, PCD
import open3d as o3d
import numpy as np
from tqdm import tqdm
import imageio.v2 as iio
import cv2
from ultralytics import YOLO
import re
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import cv2
import pickle


import math
import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union
import socket


FORCE_REFRESH = True  # True로 설정하면 캐시 무시하고 재계산



@dataclass
class RoiRow:
    roi_id: Union[int, str]
    cad_xyz: Sequence[float]          # (x,y,z)
    source_xyz: Sequence[float]       # (x,y,z)
    real_ng: bool                     # 실제 NG 여부(너의 로직 결과)
    distance_threshold: Optional[float] = None  # 거리 NG 판정 기준 (None이면 판정 안 함)

class PointCloudView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=2000)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)

        self.scatter = None
        self.result_pcd = o3d.geometry.PointCloud()
        self.result_T = np.eye(4, dtype=np.float64)
        self.T_list = []
        self.frame_idx = {}

        g = gl.GLGridItem()
        g.scale(200, 200, 1)
        self.view.addItem(g)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._cad_scale = 1.0,        
        self.setWindowTitle("Body Hole Auto Insepction System")
        self.resize(1920, 1020)
        root = QWidget()
        self.setCentralWidget(root)        
        self.utils = Utils()
        self.pcd = PCD()
        self._set_path()

        
        leftWidget = QWidget()
        rightWidget = QWidget()        
        layout = QVBoxLayout(root)
        contentLayout = QHBoxLayout()

        contentLayout.addWidget(leftWidget, 3)
        contentLayout.addWidget(rightWidget, 1)
        layout.addLayout(contentLayout)

        leftLayout = QVBoxLayout(leftWidget)
        leftLayout.setContentsMargins(0, 0, 0, 0)
        self.view3d = PointCloudView()
        leftLayout.addWidget(self.view3d)

        rightLayout = QVBoxLayout(rightWidget)
        rightLayout.setAlignment(Qt.AlignmentFlag.AlignTop)

        rightLayout.addWidget(QLabel("Model"))
        self.radioGroup = QButtonGroup(self)
        # self.radioL = QRadioButton("L Model")
        # self.radioR = QRadioButton("R Model")
        # self.radioR.setChecked(True)
        self.radioGroup.addButton(self.radioL)
        self.radioGroup.addButton(self.radioR) 
        radioRow = QHBoxLayout()
        radioRow.addWidget(self.radioL)
        radioRow.addWidget(self.radioR)
        radioRow.addStretch(1)
        rightLayout.addLayout(radioRow)

        sourceDataFolderRow = QHBoxLayout()
        sourceDataFolderRow.addWidget(QLabel("Source Data"))
        # self.tbSourceDataFolderPath = QLineEdit(rf"C:\Users\SehoonKang\Desktop\dataset\260113_Scan\260113_Scan\RH")
        self.tbSourceDataFolderPath = QLineEdit(self._source_dir)
        sourceDataFolderRow.addWidget(self.tbSourceDataFolderPath)
        self.btnSourceDataLoad = QPushButton("Load")
        sourceDataFolderRow.addWidget(self.btnSourceDataLoad)
        rightLayout.addLayout(sourceDataFolderRow)
        
        calibrationFileRow = QHBoxLayout()
        calibrationFileRow.addWidget(QLabel("Calibration File"))
        # self.tbCalibrationFilePath = QLineEdit(rf"C:\Users\SehoonKang\Desktop\dataset\260113_Scan\260113_Scan\cam_robot_extrinsic_0_1_hand_eye.yml")
        self.tbCalibrationFilePath = QLineEdit(self._calib_path)
        calibrationFileRow.addWidget(self.tbCalibrationFilePath)
        self.btnCalibrationFilePath = QPushButton("Load")
        calibrationFileRow.addWidget(self.btnCalibrationFilePath)
        rightLayout.addLayout(calibrationFileRow)

        deepLearningFileRow = QHBoxLayout()
        deepLearningFileRow.addWidget(QLabel("Deep Learning"))
        # self.tbDeepLearningModelFilePath = QLineEdit(rf"C:\Users\SehoonKang\Desktop\dataset\260113_Scan\260113_Scan\260120_seg_v2.pt")        
        self.tbDeepLearningModelFilePath = QLineEdit(self._seg_model_path)   
        deepLearningFileRow.addWidget(self.tbDeepLearningModelFilePath)
        self.btnDeepLearningFilePath = QPushButton("Load")
        deepLearningFileRow.addWidget(self.btnDeepLearningFilePath)
        rightLayout.addLayout(deepLearningFileRow)
        
        self.btnMerge = QPushButton("Merge")
        self.btnInspect = QPushButton("Inspect")
        
        rightLayout.addWidget(self.btnMerge)
        rightLayout.addWidget(self.btnInspect)
        rightLayout.addWidget(QLabel("Log"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        rightLayout.addWidget(self.log, 1)

        self.btnSourceDataLoad.clicked.connect(self.on_source_data_load)
        self.btnCalibrationFilePath.clicked.connect(self.on_calibration_file_load)
        self.btnDeepLearningFilePath.clicked.connect(self.on_deep_learning_file_load)        
        self.btnMerge.clicked.connect(self.on_merge)
        self.btnInspect.clicked.connect(self.on_inspect)


    def _set_path(self):
        config = load_cfg()
        # os 맞게 다시한번 확인
        self._source_dir = str(Path(config["source_data_folder"]))
        self._calib_path = str(Path(config["calibration_file"]))
        self._seg_model_path = str(Path(config["deep_learning_model"]))

        self.pcd.set_path(body_path=str(Path(config["bodyPath"])))
        body_pose = str(config["BodyPosition"])

        self.radioL = QRadioButton("L Model")
        self.radioR = QRadioButton("R Model")
        if body_pose == "Left":
            self.radioL.setChecked(True)
        elif body_pose == "Right":
            self.radioR.setChecked(True)
        else:
            print(f"Body Pose UnKnown Left or Right yout Input: {body_pose}")
            print("set Right Pose default")
            self.radioR.setChecked(True)
        self._cad_scale = float(config["cad_scale"]),   
        

    def on_source_data_load(self):        
        self.utils.on_load_source_data_folder(self.tbSourceDataFolderPath.text(), FileType.Image)
        self.log.append(rf"load {self.tbSourceDataFolderPath.text()} completed.")

    def on_calibration_file_load(self):
        self.utils.on_load_source_data_folder(self.tbCalibrationFilePath.text(), FileType.Calibration)
        self.log.append(rf"load {self.tbCalibrationFilePath.text()} completed.")

    def on_deep_learning_file_load(self):
        self.utils.on_load_source_data_folder(self.tbDeepLearningModelFilePath.text(), FileType.DeepLearningModel)
        self.log.append(rf"load {self.tbDeepLearningModelFilePath.text()} completed.")
        #변경 필요 ========================================
        self.seg_model = YOLO(self.tbDeepLearningModelFilePath.text())

    def on_merge(self):
        self.log.append("[INFO] Start to merge frames")


        # load Cache Start
        cache_file = f"cache/merge_pcd_{self.current_model()}.pkl"

        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = f"{cache_dir}/merge_pcd_{self.current_model()}.pkl"
        merged_pcd_file = f"{cache_dir}/merged_pcd_{self.current_model()}.ply"

        
        if not FORCE_REFRESH and os.path.exists(cache_file):
            print("Loading from cache...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            T_list = cached_data['T_list']
            reference_pcd = cached_data['reference_pcd']
            merged_pcd = o3d.io.read_point_cloud(merged_pcd_file)
        else:
            print("No Cache Runnign merge_pcd func")
            T_list, merged_pcd, reference_pcd = self.pcd.merge_pcd(
                self.utils.source_data_folder_files,
                self.utils.calibration_file_path,
                "fanuc", 
                self.current_model()
            )
            
            # 캐시 저장
            print("Saving to cache...")
            o3d.io.write_point_cloud(merged_pcd_file, merged_pcd)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'T_list': T_list,
                    'reference_pcd': reference_pcd
                }, f)
                
        #load Cache End

            # T_list, merged_pcd, reference_pcd = self.pcd.merge_pcd(self.utils.source_data_folder_files,
            #                                                     self.utils.calibration_file_path,
            #                                                     "fanuc", self.current_model())
        

        # Test Cached_data



        self.T_list = T_list
        cad_centers_array = np.array(self.utils.cad_data[self.current_model()]["cad_centers"], dtype=np.float32)

        moved_merge_pcd, T_to_cad, report = self.pcd.move_merged_pcd_to_cad(merged_pcd=merged_pcd,
                                                                            CAD_CENTERS=cad_centers_array,
                                                                            align_points=np.asarray(reference_pcd, dtype=np.float64),
                                                                            copy_pcd=True)
        
        self.result_pcd = moved_merge_pcd
        self.result_T = T_to_cad

        pcd_base = copy.deepcopy(self.result_pcd)   # 또는 self.result_pcd.voxel_down_sample(...)

        cad_points = np.array(self.utils.cad_data[self.current_model()]["cad_welding_points"], dtype=np.float32)
        
        # # cad point scale 곱하기
        # cad_points = cad_points * self._cad_scale 


        pcd_cad = o3d.geometry.PointCloud()
        pcd_cad.points = o3d.utility.Vector3dVector(cad_points.astype(np.float64))
        pcd_cad.paint_uniform_color((1.0, 0.0, 0.0))

        o3d.visualization.draw_geometries(
            [pcd_base, pcd_cad],
            window_name="base + cad_points",
            point_show_normal=False
        )

        self.set_pointcloud(moved_merge_pcd)
        self.log.append("merge frames complete.")

    def on_inspect(self):
        self.log.append("Inspecting data...")
        roi_hole_points_dict = {}        

        def extract_index(group):
            fname = os.path.basename(group[0])
            m = re.match(r"(\d+)_", fname)
            return int(m.group(1)) if m else -1

        source_data_folder_files_sort = sorted(self.utils.source_data_folder_files, key=extract_index)

        frame_pcd = {}
        pose_dict = {}
        self.frame_idx = {} 

        for i, frame in enumerate(tqdm(source_data_folder_files_sort, total=len(source_data_folder_files_sort))):
            pcd = PCD()
            texture_path, x_path, y_path, z_path, pose_path, mask_path = frame

            pts_cam = pcd._make_cam_pcd(x_path=x_path, y_path=y_path, z_path=z_path,texture_path=texture_path, mask_path=mask_path)
            frame_number = os.path.basename(texture_path).replace("_IMG_Texture_8Bit.png", "")
            X = np.asarray(iio.imread(x_path).astype(np.float64))
            Y = np.asarray(iio.imread(y_path).astype(np.float64))
            Z = np.asarray(iio.imread(z_path).astype(np.float64))

            mask_valid = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
            mask_valid = np.asarray(mask_valid, dtype=bool)
            mask_nonzero = (X != 0) | (Y != 0) | (Z != 0)
            mask_nonzero = np.asarray(mask_nonzero, dtype=bool)
            mask_valid &= mask_nonzero

            ys_idx, xs_idx = np.where(mask_valid)
            pts_cam = np.stack([
                X[ys_idx, xs_idx],
                Y[ys_idx, xs_idx],
                Z[ys_idx, xs_idx]
            ], axis=1)            
            
            T_cam_to_world = self.T_list[i]
            T_world_to_cad = self.result_T
            T_cam_to_cad = T_world_to_cad @ T_cam_to_world
            self.frame_idx[frame_number] = i

            pts_cam = self.transform_points(pts_cam, T_cam_to_world)   
            pts_cam = self.transform_points(pts_cam, T_world_to_cad)
            frame_pcd[frame_number] = pts_cam

            image_for_seg = cv2.imread(texture_path, cv2.IMREAD_COLOR)
            img_h, img_w, _ = image_for_seg.shape

            cad_points  = np.array(self.utils.cad_data[self.current_model()]["cad_welding_points"], dtype=np.float32)            
            pcd_cad = o3d.geometry.PointCloud()
            pcd_cad.points = o3d.utility.Vector3dVector(cad_points.astype(np.float64))
            
            pp = np.asarray(pts_cam, dtype=np.float64).reshape(-1, 3)
            pcd_pts_cad = o3d.geometry.PointCloud()
            pcd_pts_cad.points = o3d.utility.Vector3dVector(pp)

            seg_pad = 150

            for roi_id, center in enumerate(cad_points, start=1):
                dist = np.linalg.norm(pts_cam - center, axis=1)                
                mask_roi_3d = dist <= 8
                num_roi_pts = np.count_nonzero(mask_roi_3d)

                if num_roi_pts == 0:
                    print("num_roi_pts == 0")
                    continue

                roi_y = ys_idx[mask_roi_3d]
                roi_x = xs_idx[mask_roi_3d]

                y_min, y_max = int(roi_y.min()), int(roi_y.max())
                x_min, x_max = int(roi_x.min()), int(roi_x.max())

                pad = seg_pad
                y_min = max(y_min - pad, 0)
                x_min = max(x_min - pad, 0)
                y_max = min(y_max + pad, img_h - 1)
                x_max = min(x_max + pad, img_w - 1)

                if y_max <= y_min or x_max <= x_min:
                    print(rf"{frame_number} // {roi_id} : ymx_{y_max} / ymn_{y_min} / xmx_{x_max} / xmn_{x_min}")
                    continue

                crop_img = image_for_seg[y_min:y_max + 1, x_min:x_max + 1]

                if crop_img.size == 0:                    
                    continue
                
                ch, cw, _ = crop_img.shape
                if ch < 16 or cw < 16:                    
                    continue

                results = self.seg_model(crop_img, device='cuda:0', verbose=False)

                if len(results) == 0 or results[0].masks is None:
                    continue

                masks_yolo = results[0].masks.data.cpu().numpy()
                if masks_yolo.shape[0] == 0:
                    print(f"[INFO] ROI : mask 개수 0 (view {i}).")
                    continue

                mask_bin = (masks_yolo > 0.5)
                full_mask_local = np.any(mask_bin, axis=0)
                Hm, Wm = full_mask_local.shape

                if (Hm, Wm) != (ch, cw):
                    full_mask_local = cv2.resize(full_mask_local.astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST).astype(bool)                 
                
                mask_resized = full_mask_local
                in_crop = (
                    (ys_idx >= y_min) & (ys_idx <= y_max) &
                    (xs_idx >= x_min) & (xs_idx <= x_max)
                )
                if not np.any(in_crop):
                    continue

                ys_c = ys_idx[in_crop] - y_min
                xs_c = xs_idx[in_crop] - x_min

                mask_on_pixels_small = mask_resized[ys_c, xs_c]
                mask_hole_small = mask_on_pixels_small & mask_roi_3d[in_crop]

                pcd_pts_np = np.asarray(pcd_pts_cad.points) 
                roi_hole_pts = pcd_pts_np[in_crop][mask_hole_small]
                n_hole = roi_hole_pts.shape[0]
                
                if n_hole > 0:
                    roi_hole_points_dict.setdefault(roi_id, {})[frame_number] = roi_hole_pts


            pose = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', open(pose_path, 'r', encoding='utf-8').read())]
            pose = pose[:6]
            pose_dict[frame_number] = pose

        self.inspect_real_welding_point(roi_hole_points_dict=roi_hole_points_dict, frame_pcd=frame_pcd, pad=5)
        # pcd_base = copy.deepcopy(self.result_pcd)        
        # pcd_base = self.result_pcd.voxel_down_sample(0.5)

        # pcd_holes = self.roi_dict_to_pcd(roi_hole_points_dict=roi_hole_points_dict)

        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(pcd_base)
        # vis.add_geometry(pcd_holes)        
        # opt = vis.get_render_option()
        # opt.point_size = 4.0
        # vis.run()
        # vis.destroy_window()

        # self.set_pointcloud(pcd_holes)

    def set_pointcloud(self, pcd: o3d.geometry.PointCloud, *, size: float = 15.0, sampling_rate : 0.5):
        pcd = pcd.voxel_down_sample(sampling_rate)
        pts = np.asarray(pcd.points, dtype=np.float32)

        if pcd.has_colors():
            cols = np.asarray(pcd.colors, dtype=np.float32)
            if cols.max() > 1.0:
                cols = cols / 255.0
            alpha = np.ones((cols.shape[0], 1), dtype=np.float32)
            cols = np.concatenate([cols, alpha], axis=1)
        else:
            cols = np.ones((pts.shape[0], 4), dtype=np.float32) * 0.8
            cols[:, 3] = 1.0

        # ✅ 기존 scatter 제거 (중복 렌더 방지)
        if getattr(self.view3d, "scatter", None) is not None:
            self.view3d.view.removeItem(self.view3d.scatter)
            self.view3d.scatter = None

        # ✅ 점 크기 키우기
        self.view3d.scatter = gl.GLScatterPlotItem(pos=pts, color=cols, size=float(size), pxMode=True)
        self.view3d.view.addItem(self.view3d.scatter)

    def roi_dict_to_pcd(self, roi_hole_points_dict: dict[int, dict[int, np.ndarray]]) -> o3d.geometry.PointCloud:
        all_pts = []
        
        # 첫 번째 dict: roi_id, 두 번째 dict: 내부 index 또는 key
        for roi_id, hole_dict in roi_hole_points_dict.items():
            for frame_id, pts in hole_dict.items(): # 리스트 대신 딕셔너리 순회
                if pts is None or len(pts) == 0:
                    print(rf">>>>>>> point count is zero : {roi_id}_{frame_id}")
                    continue
                
                # pts가 이미 (N, 3) 형태의 numpy array라고 가정합니다.
                all_pts.append(np.asarray(pts, dtype=np.float64))

        if not all_pts:
            return o3d.geometry.PointCloud()

        # 모든 포인트를 하나로 합칩니다.
        P = np.vstack(all_pts)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(P)
        
        # 가시성을 위해 색상 지정 (연두색)
        pcd.paint_uniform_color([0.0, 1.0, 0.2])
        return pcd    

    def current_model(self):        
        return 'LH' if self.radioL.isChecked() else 'RH'
    
    def set_pointcloud(self, pcd: o3d.geometry.PointCloud):
            pts = np.asarray(pcd.points, dtype=np.float32)
            if pcd.has_colors():
                cols = np.asarray(pcd.colors, dtype=np.float32)

                if cols.max() > 1.0:
                    cols = cols / 255.0

                alpha = np.ones((cols.shape[0], 1), dtype=np.float32)
                cols = np.concatenate([cols, alpha], axis=1)
            else:                
                cols = np.ones((pts.shape[0], 4), dtype=np.float32) * 0.8
                cols[:, 3] = 1.0

            self.view3d.scatter = gl.GLScatterPlotItem(pos=pts, color=cols, size=2.0, pxMode=True)
            self.view3d.view.addItem(self.view3d.scatter)

    def transform_points(self, points_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
        points_xyz = np.asarray(points_xyz, dtype=np.float64)
        if points_xyz.ndim == 1:
            points_xyz = points_xyz.reshape(1, 3)

        ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
        ph = np.hstack([points_xyz, ones])            # (N,4)
        out = (T @ ph.T).T[:, :3]                     # (N,3)
        return out
    
    def inspect_real_welding_point(self,                                 
                                   roi_hole_points_dict : dict[int, dict[int, np.ndarray]],
                                   frame_pcd : dict[int, o3d.geometry.PointCloud],
                                   pad: int = 30) : 
        samples = []        
        for roi_id, pcd_dict in roi_hole_points_dict.items():
            print(rf"{roi_id+1}번째 타흔 확인")

            # 1. frame 별로 가장 point cloud 수가 많은 것의
            if not pcd_dict:
                print("  [WARN] pcd_dict empty")
                continue
            best_frame_id, best_roi_pts = max(
                pcd_dict.items(),
                key=lambda kv: 0 if kv[1] is None else int(np.asarray(kv[1]).shape[0])
            )

            best_frame_pcd = frame_pcd.get(best_frame_id, None)
            if best_frame_pcd is None:
                print(f"[ROI {roi_id}] best_frame_pcd is None (frame_id={best_frame_id})")
                continue
            
            if isinstance(best_frame_pcd, o3d.geometry.PointCloud):
                frame_pts = np.asarray(best_frame_pcd.points, dtype=np.float64)
                frame_pcd_o3d = best_frame_pcd
            else:
                # best_frame_pcd가 np.ndarray라고 가정 (N,3)
                frame_pts = np.asarray(best_frame_pcd, dtype=np.float64).reshape(-1, 3)
                frame_pcd_o3d = o3d.geometry.PointCloud()
                frame_pcd_o3d.points = o3d.utility.Vector3dVector(frame_pts)

            # 2. 평면 피팅
            center = np.asarray(best_roi_pts, dtype=np.float64).mean(axis=0)            
            dist = np.linalg.norm(frame_pts - center[None, :], axis=1)
            crop_idx = np.where(dist <= float(pad))[0]

            if crop_idx.size == 0:
                print(f"[ROI {roi_id}] crop empty (pad={pad})")
                continue

            pcd_crop = frame_pcd_o3d.select_by_index(crop_idx.tolist())
            pcd_filtered, inlier_idx = pcd_crop.remove_radius_outlier(
                nb_points=30,
                radius=1.0
            )

            pcd_near, pcd_far, plane, is_welding, w_xyz = self.filter_points_near_plane(pcd_filtered,
                                                                     distance_threshold=0.05,
                                                                     frame_idx = self.frame_idx[best_frame_id],
                                                                     center=center,
                                                                     min_max_threshold = 0.2)
            
            if w_xyz is None or np.asarray(w_xyz).size != 3:
                print(f"[ROI {roi_id}] w_xyz invalid: {w_xyz} (is_welding={is_welding})")
                continue

            cad_center_point = self.utils.cad_data[self.current_model()]["cad_welding_points"][roi_id - 1]
            cad_center_point = np.asarray(cad_center_point, dtype=np.float64).reshape(3)
            source_center_point = np.asarray(w_xyz, dtype=np.float64).reshape(3)

            distance = self.distance_3d(cad_center_point, source_center_point)

            def sphere_at(p, radius=2.0, color=(1.0, 0.0, 0.0)):
                p = np.asarray(p, dtype=np.float64).reshape(3)
                s = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius))
                s.translate(p)
                s.paint_uniform_color(color)
                return s

            # w_xyz: np.array([w_x, w_y, w_z])
            # cad_center_point: (3,) array-like

            # w_sphere   = sphere_at(w_xyz, radius=0.5, color=(0.5, 0.5, 0.5))   # 빨강: w_xyz
            # cad_sphere = sphere_at(cad_center_point, radius=0.5, color=(0.0, 0, 1)) 

            # print(rf"Welding Point {roi_id} : CAD X : {cad_center_point[0]} / CAD Y : {cad_center_point[1]} / CAD Z : {cad_center_point[2]}")
            # print(rf"Welding Point {roi_id} : SRC X : {source_center_point[0]} / SRC Y : {source_center_point[1]} / SRC Z : {source_center_point[2]}")
            # print(rf"Welding Point {roi_id} : DIST : {distance}")
            # samples.append(RoiRow(roi_id=roi_id, cad_xyz=(cad_center_point[0], cad_center_point[1], cad_center_point[2]), source_xyz=(source_center_point[0], source_center_point[1], source_center_point[2]), real_ng=is_welding, distance_threshold= 4))
            # pcd_near.paint_uniform_color((0,1,0))
            # pcd_far.paint_uniform_color((1, 0, 0))
            # o3d.visualization.draw_geometries([pcd_near, pcd_far, cad_sphere, w_sphere])
            
        samples.sort(key=lambda r: r.roi_id)
        # self.export_roi_distance_excel(samples, rf"{self.current_model()}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        self.export_roi_distance_excel(samples, rf"{self.current_model()}_report.xlsx")
            # print(rf">>>>>>>>>>>>>>>>>> {best_frame_id} // {roi_id} // {is_welding} // {w_xyz}")

            # s = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
            # s.translate(center)
            # s.paint_uniform_color((1.0, 0.0, 0.0))  # center = red
            
            # pcd_hole = o3d.geometry.PointCloud()
            # pcd_hole.points = o3d.utility.Vector3dVector(pts_cat.astype(np.float64))
            # pcd_hole.paint_uniform_color((0.0, 0.0, 1.0))

            # o3d.visualization.draw_geometries([pcd_filtered, pcd_hole, s],window_name=f"ROI {roi_id} crop (pad={pad})")

    def filter_points_near_plane(self, pcd, distance_threshold=1.0, ransac_thresh=1.0,
                             ransac_n=3, num_iterations=2000,
                             frame_idx: int | None = None,
                             center: np.ndarray | None = None,
                             trim_ratio: float = 0.05,
                             count_threshold: int = 30,
                             min_max_threshold: float = 0.2):

        pts = np.asarray(pcd.points, dtype=np.float64)
        if pts.size == 0:
            return None, None, np.array([], dtype=np.int64), False, None

        # 1) plane fit
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=float(ransac_thresh),
            ransac_n=int(ransac_n),
            num_iterations=int(num_iterations),
        )
        a, b, c, d = plane_model
        n = np.array([a, b, c], dtype=np.float64)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            raise RuntimeError("Plane normal is near zero.")

        # 2) 법선 부호 고정 (cam -> object)
        if frame_idx is not None and center is not None:
            T_cam_to_world = self.T_list[frame_idx]
            T_cam_to_cad = self.result_T @ T_cam_to_world
            cam_pos_cad = T_cam_to_cad[:3, 3]

            center = np.asarray(center, dtype=np.float64).reshape(3)
            v_ref = center - cam_pos_cad  # cam -> object (필요하면 반대로 바꿔 테스트)

            if np.dot(n, v_ref) < 0:
                n = -n
                d = -d
                plane_model = [n[0], n[1], n[2], d]

        # 3) signed distance
        n_norm = (np.linalg.norm(n) + 1e-12)
        signed = (pts @ n + d) / n_norm

        keep = signed > float(distance_threshold)
        
        keep_idx = np.where(keep)[0]
        rem_idx  = np.where(~keep)[0]

        def trim_indices_by_signed(global_idx: np.ndarray, signed_all: np.ndarray, trim_ratio: float) -> np.ndarray:
            """global_idx에 해당하는 signed 값에서 상/하위 trim_ratio 제거 후 남는 global_idx 반환"""
            if global_idx.size == 0:
                return global_idx

            s = signed_all[global_idx]  # 해당 그룹의 signed 값들

            lo = np.percentile(s, 100.0 * trim_ratio)
            hi = np.percentile(s, 100.0 * (1.0 - trim_ratio))

            keep_local = (s >= lo) & (s <= hi)
            return global_idx[keep_local]

        trim_ratio = float(trim_ratio)  # 파라미터로 받고 있다고 가정

        keep_idx_trim = trim_indices_by_signed(keep_idx, signed, trim_ratio)
        rem_idx_trim  = trim_indices_by_signed(rem_idx,  signed, trim_ratio)

        below = pcd.select_by_index(keep_idx_trim.tolist())
        above = pcd.select_by_index(rem_idx_trim.tolist())

        # 1. above filtering
        above = self.remove_nearest_percent_from_above(above, below, 30)
        above.paint_uniform_color((1, 0, 0))

        # 2. above plane fitting
        plane_model2, n2, above_inlier, above_outlier = self.fit_plane_from_pcd(above, ransac_thresh=0.05)

        # 법선 부호 고정 (cam -> object)
        a2, b2, c2, d2 = plane_model2
        n2 = np.array([a2, b2, c2], dtype=np.float64)

        if frame_idx is not None and center is not None:
            T_cam_to_world = self.T_list[frame_idx]
            T_cam_to_cad = self.result_T @ T_cam_to_world
            cam_pos_cad = T_cam_to_cad[:3, 3]

            center = np.asarray(center, dtype=np.float64).reshape(3)
            v_ref = center - cam_pos_cad  # cam -> object (필요하면 반대로 바꿔 테스트)

            if np.dot(n2, v_ref) < 0:
                n2 = -n2
                d2 = -d2
                plane_model2 = [float(n2[0]), float(n2[1]), float(n2[2]), float(d2)]

        # 3. divide by plane2 
        a2, b2, c2, d2 = plane_model2
        n2 = np.array([a2, b2, c2], dtype=np.float64)
        n2_norm = np.linalg.norm(n2) + 1e-12

        signed2 = (pts @ n2 + d2) / n2_norm
        near_thr2 = 0.03
        near_mask = signed2 > float(near_thr2)
        far_mask  = ~near_mask

        near_idx = np.where(near_mask)[0]
        far_idx  = np.where(far_mask)[0]

        welding_pcd = pcd.select_by_index(near_idx.tolist())
        plane_pcd  = pcd.select_by_index(far_idx.tolist())

        # eps = self.auto_eps(welding_pcd, factor=2.5)
        # welding_pcd = self.keep_largest_spatial_component(welding_pcd, eps, min_points=10) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! min_points로 클러스터링 최적화 가능

        if len(welding_pcd.points) == 0:
            return welding_pcd, plane_pcd, plane_model2, False, None

        # welding_pcd.paint_uniform_color([1, 0, 0]) 
        # plane_pcd.paint_uniform_color([0, 1, 0])
        # o3d.visualization.draw_geometries([
        #     welding_pcd,
        #     plane_pcd
        # ])

    #     # 4. confirm real welding points 
        n_welding = len(welding_pcd.points)
        n_plane   = len(plane_pcd.points)

        if n_welding < count_threshold or n_plane < count_threshold:
            return welding_pcd, plane_pcd, plane_model2, False, None

    #     # plane1 기준 signed를 welding/plane 각각 다시 계산
        w_pts = np.asarray(welding_pcd.points, dtype=np.float64)
        p_pts = np.asarray(plane_pcd.points, dtype=np.float64)

        s_welding = (w_pts @ n2 + d2) / n2_norm
        s_plane   = (p_pts @ n2 + d2) / n2_norm

        hi_a = np.percentile(s_plane, 100.0 * (1.0 - trim_ratio))
        s_plane_trim = s_plane[s_plane <= hi_a]

        lo_b = np.percentile(s_welding, 100.0 * trim_ratio)
        s_welding_trim = s_welding[s_welding >= lo_b]

        plane_min = float(s_plane_trim.min())
        welding_max = float(s_welding_trim.max())
        gap = welding_max - plane_min

        print(rf">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> gap = {gap}")
        is_welding = gap >= float(min_max_threshold)

        # 5. center point
        if w_pts.size == 0:
            return welding_pcd, plane_pcd, plane_model2, False, None

        w_center = w_pts.mean(axis=0)
        w_x, w_y = float(w_center[0]), float(w_center[1])

        a2, b2, c2, d2 = plane_model2
        if abs(c2) < 1e-12:
            return welding_pcd, plane_pcd, plane_model2, is_welding, None

        w_z = -(a2 * w_x + b2 * w_y + d2) / c2
        w_xyz = np.array([w_x, w_y, w_z], dtype=np.float64)

        return welding_pcd, plane_pcd, plane_model, is_welding, w_center
    
    def fit_plane_from_pcd(self, pcd: o3d.geometry.PointCloud,
                       ransac_thresh: float = 0.05,
                       ransac_n: int = 3,
                       num_iterations: int = 2000):
        if pcd is None or len(pcd.points) < ransac_n:
            raise ValueError("Not enough points to fit a plane.")

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=float(ransac_thresh),
            ransac_n=int(ransac_n),
            num_iterations=int(num_iterations),
        )
        a, b, c, d = plane_model
        n = np.array([a, b, c], dtype=np.float64)
        n = n / (np.linalg.norm(n) + 1e-12)

        pcd_inlier = pcd.select_by_index(inliers)
        pcd_outlier = pcd.select_by_index(inliers, invert=True)

        return plane_model, n, pcd_inlier, pcd_outlier    
    
    def remove_nearest_percent_from_above(
        self,
        above: o3d.geometry.PointCloud,
        below: o3d.geometry.PointCloud,
        n_percent: float = 20.0  # 가까운 n% 제거
    ) -> o3d.geometry.PointCloud:
        A = np.asarray(above.points, dtype=np.float64)
        B = np.asarray(below.points, dtype=np.float64)

        if A.size == 0:
            return o3d.geometry.PointCloud()
        if B.size == 0:
            # below가 비어있으면 기준점이 없으니 above 그대로
            return above

        # 1) below 평균점(중심)
        c = B.mean(axis=0)  # (3,)

        # 2) above 각 점의 중심까지 거리
        dist = np.linalg.norm(A - c[None, :], axis=1)

        # 3) 가까운 n% 인덱스 계산
        n_percent = float(n_percent)
        n_percent = max(0.0, min(100.0, n_percent))
        k = int(np.floor(len(dist) * (n_percent / 100.0)))

        if k <= 0:
            return above  # 제거할 게 없음
        if k >= len(dist):
            return o3d.geometry.PointCloud()  # 전부 제거

        order = np.argsort(dist)      # 가까운 순
        remove_idx = order[:k]        # 가까운 k개 제거

        # 4) 제거 후 남길 인덱스
        keep_mask = np.ones(len(dist), dtype=bool)
        keep_mask[remove_idx] = False
        keep_idx = np.where(keep_mask)[0]

        return above.select_by_index(keep_idx.tolist())
    
    def keep_largest_spatial_component(self, pcd: o3d.geometry.PointCloud, eps: float = 2.0, min_points: int = 10):
        
        if pcd is None or len(pcd.points) == 0:
            return o3d.geometry.PointCloud()

        labels = np.asarray(pcd.cluster_dbscan(eps=float(eps), min_points=int(min_points), print_progress=False), dtype=np.int32)

        print("N:", len(labels), "noise:", np.sum(labels == -1), "clusters:", len(set(labels)) - (1 if -1 in labels else 0))

        # 전부 노이즈(-1)면 빈 결과
        valid = labels >= 0
        if not np.any(valid):
            return o3d.geometry.PointCloud()

        counts = np.bincount(labels[valid])
        largest_label = int(np.argmax(counts))

        keep_idx = np.where(labels == largest_label)[0]
        return pcd.select_by_index(keep_idx.tolist())
    
    def auto_eps(self, pcd: o3d.geometry.PointCloud, factor: float = 2.5, sample: int = 5000) -> float:
        if len(pcd.points) == 0:
            return 1.0
        if len(pcd.points) > sample:
            idx = np.random.choice(len(pcd.points), sample, replace=False)
            p = pcd.select_by_index(idx.tolist())
        else:
            p = pcd
        d = np.asarray(p.compute_nearest_neighbor_distance())
        if d.size == 0:
            return 1.0
        return float(np.median(d) * factor)

    def distance_3d(self, p1, p2):
        p1 = np.asarray(p1, dtype=np.float64).reshape(3)
        p2 = np.asarray(p2, dtype=np.float64).reshape(3)
        return float(np.linalg.norm(p1 - p2))

    def export_roi_distance_excel(
        self,
        rows: Iterable[RoiRow],
        out_xlsx_path: str,
        sheet_name: str = "ROI_Report",
    ) -> str:
        """
        ROI별 CAD 좌표, Source 좌표, 거리, NG 판단을 엑셀로 저장.
        distance = Euclidean distance between cad_xyz and source_xyz.
        distance_ng = (distance > distance_threshold) if threshold provided else None

        returns: saved xlsx path
        """
        headers = [
            "Welding ID",
            "Cad x", "Cad y", "Cad z",
            "Real x", "Real y", "Real z",
            "Distance",
            "Real Welding",
            "Distance OK",
        ]

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = sheet_name

        # Header styling
        header_fill = PatternFill("solid", fgColor="1F4E79")  # dark blue
        header_font = Font(color="FFFFFF", bold=True)
        center = Alignment(horizontal="center", vertical="center")

        ws.append(headers)
        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center

        # Data rows
        red_fill = PatternFill("solid", fgColor="FFC7CE")     # light red
        red_font = Font(color="9C0006", bold=True)

        def _dist(a, b) -> float:
            dx = float(a[0]) - float(b[0])
            dy = float(a[1]) - float(b[1])
            dz = float(a[2]) - float(b[2])
            return math.sqrt(dx*dx + dy*dy + dz*dz)

        r = 2
        for item in rows:
            cad = item.cad_xyz
            src = item.source_xyz
            dist = _dist(cad, src)

            if item.distance_threshold is None:
                dist_ng = ""  # 기준 없으면 공란
            else:
                dist_ng = bool(dist < float(item.distance_threshold))

            ws.append([
                str(item.roi_id),
                float(cad[0]), float(cad[1]), float(cad[2]),
                float(src[0]), float(src[1]), float(src[2]),
                float(dist),
                bool(item.real_ng),
                dist_ng if dist_ng != "" else "",
            ])

            # Align + number formats
            for c in range(1, len(headers) + 1):
                ws.cell(row=r, column=c).alignment = center

            # 좌표/거리 소수점 포맷
            for c in range(2, 9):  # cad/source/distance
                ws.cell(row=r, column=c).number_format = "0.000"

            # NG 강조(둘 중 하나라도 True면 행을 붉게)
            real_ng_val = bool(item.real_ng)
            dist_ng_val = bool(dist_ng) if dist_ng != "" else False
            if real_ng_val == False or dist_ng_val == False:
                for c in range(1, len(headers) + 1):
                    cell = ws.cell(row=r, column=c)
                    cell.fill = red_fill
                    # 글씨도 강조하고 싶으면:
                    if c in (9, 10):  # real ng, distance ng
                        cell.font = red_font

            r += 1

        # Column width autosize (simple)
        for col_idx in range(1, len(headers) + 1):
            col_letter = get_column_letter(col_idx)
            max_len = 0
            for cell in ws[col_letter]:
                val = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, len(val))
            ws.column_dimensions[col_letter].width = min(max_len + 2, 28)

        ws.freeze_panes = "A2"
        wb.save(out_xlsx_path)
        return out_xlsx_path

def load_cfg():

    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        all_configs = json.load(f)
    
    # 현재 컴퓨터 이름 가져오기
    hostname = socket.gethostname()

    print(f"Computer Name : {hostname}")
    
    # 해당 컴퓨터의 설정 가져오기 (없으면 default 사용)
    config = all_configs.get(hostname, all_configs.get("default", {}))
    
    print(f"Loading config for: {hostname}")
    return config



def main():
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":    
    main()