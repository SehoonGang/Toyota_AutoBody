import json
import os
import sys
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

import matplotlib.pyplot as plt
import cv2

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

        g = gl.GLGridItem()
        g.scale(200, 200, 1)
        self.view.addItem(g)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()        
        self.setWindowTitle("Body Hole Auto Insepction System")
        self.resize(1920, 1020)
        root = QWidget()
        self.setCentralWidget(root)        
        self.utils = Utils()
        self.pcd = PCD()
        
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
        self.radioL = QRadioButton("L Model")
        self.radioR = QRadioButton("R Model")
        self.radioL.setChecked(True)
        self.radioGroup.addButton(self.radioL)
        self.radioGroup.addButton(self.radioR) 
        radioRow = QHBoxLayout()
        radioRow.addWidget(self.radioL)
        radioRow.addWidget(self.radioR)
        radioRow.addStretch(1)
        rightLayout.addLayout(radioRow)

        sourceDataFolderRow = QHBoxLayout()
        sourceDataFolderRow.addWidget(QLabel("Source Data"))
        self.tbSourceDataFolderPath = QLineEdit(rf"C:\Users\SehoonKang\Desktop\dataset\260113_Scan\260113_Scan\LH")
        sourceDataFolderRow.addWidget(self.tbSourceDataFolderPath)
        self.btnSourceDataLoad = QPushButton("Load")
        sourceDataFolderRow.addWidget(self.btnSourceDataLoad)
        rightLayout.addLayout(sourceDataFolderRow)
        
        calibrationFileRow = QHBoxLayout()
        calibrationFileRow.addWidget(QLabel("Calibration File"))
        self.tbCalibrationFilePath = QLineEdit(rf"C:\Users\SehoonKang\Desktop\dataset\260113_Scan\260113_Scan\cam_robot_extrinsic_0_1_hand_eye.yml")
        calibrationFileRow.addWidget(self.tbCalibrationFilePath)
        self.btnCalibrationFilePath = QPushButton("Load")
        calibrationFileRow.addWidget(self.btnCalibrationFilePath)
        rightLayout.addLayout(calibrationFileRow)

        deepLearningFileRow = QHBoxLayout()
        deepLearningFileRow.addWidget(QLabel("Deep Learning"))
        self.tbDeepLearningModelFilePath = QLineEdit(rf"C:\Users\SehoonKang\Desktop\dataset\260113_Scan\260113_Scan\260120_seg_v2.pt")        
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
        T_list, merged_pcd, reference_pcd = self.pcd.merge_pcd(self.utils.source_data_folder_files,
                                                               self.utils.calibration_file_path,
                                                               "fanuc", self.current_model())
        self.T_list = T_list
        cad_centers_array = np.array(self.utils.cad_data[self.current_model()]["cad_centers"], dtype=np.float32)

        moved_merge_pcd, T_to_cad, report = self.pcd.move_merged_pcd_to_cad(merged_pcd=merged_pcd,
                                                                            CAD_CENTERS=cad_centers_array,
                                                                            align_points=np.asarray(reference_pcd, dtype=np.float64),
                                                                            copy_pcd=True)        
        
        self.result_pcd = moved_merge_pcd
        self.result_T = T_to_cad

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

            pts_cam = self.transform_points(pts_cam, T_cam_to_world)   
            pts_cam = self.transform_points(pts_cam, T_world_to_cad)            

            image_for_seg = cv2.imread(texture_path, cv2.IMREAD_COLOR)
            img_h, img_w, _ = image_for_seg.shape

            cad_points  = np.array(self.utils.cad_data[self.current_model()]["cad_welding_points"], dtype=np.float32)            
            pcd_cad = o3d.geometry.PointCloud()
            pcd_cad.points = o3d.utility.Vector3dVector(cad_points.astype(np.float64))
            
            pp = np.asarray(pts_cam, dtype=np.float64).reshape(-1, 3)
            pcd_pts_cad = o3d.geometry.PointCloud()
            pcd_pts_cad.points = o3d.utility.Vector3dVector(pp)

            for roi_id, center in enumerate(cad_points, start=1):
                dist = np.linalg.norm(pts_cam - center, axis=1)

                mask_roi_3d = dist <= 4
                num_roi_pts = np.count_nonzero(mask_roi_3d)

                if num_roi_pts == 0:
                    print("num_roi_pts == 0")
                    continue

                roi_y = ys_idx[mask_roi_3d]
                roi_x = xs_idx[mask_roi_3d]

                y_min, y_max = int(roi_y.min()), int(roi_y.max())
                x_min, x_max = int(roi_x.min()), int(roi_x.max())

                pad = 200
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

                mask_bin = (masks_yolo > 0.8)
                full_mask_local = np.any(mask_bin, axis=0)                
                Hm, Wm = full_mask_local.shape

                if (Hm, Wm) != (ch, cw):
                    full_mask_local = cv2.resize(full_mask_local.astype(np.uint8), (cw, ch),
                                                interpolation=cv2.INTER_NEAREST).astype(bool)                 
                
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
                    roi_hole_points_dict.setdefault(roi_id, []).append(roi_hole_pts)        

        self.inspect_real_welding_point(pcd_base = self.result_pcd, roi_hole_points_dict=roi_hole_points_dict, pad=5)
        pcd_base = copy.deepcopy(self.result_pcd)        
        pcd_base = self.result_pcd.voxel_down_sample(0.5)

        pcd_holes = self.roi_dict_to_pcd(roi_hole_points_dict=roi_hole_points_dict)
        pcd_holes.paint_uniform_color([1.0, 0.0, 0.0])

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd_base)
        vis.add_geometry(pcd_holes)        
        opt = vis.get_render_option()
        opt.point_size = 4.0
        vis.run()
        vis.destroy_window()

        self.set_pointcloud(pcd_holes)

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

    def roi_dict_to_pcd(self, roi_hole_points_dict: dict[int, list[np.ndarray]]) -> o3d.geometry.PointCloud:
        all_pts = []
        for roi_id, chunks in roi_hole_points_dict.items():
            for pts in chunks:
                if pts is None or len(pts) == 0:
                    continue
                all_pts.append(np.asarray(pts, dtype=np.float64))

        if not all_pts:
            return o3d.geometry.PointCloud()

        P = np.vstack(all_pts)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(P)
        
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
                                   pcd_base : o3d.geometry.PointCloud,
                                   roi_hole_points_dict : dict[int, list[np.ndarray]],
                                   pad: int = 30) :
        base_pts = np.asarray(pcd_base.points, dtype=np.float64)
        if base_pts.size == 0:
            print("[WARN] pcd_base is empty.")
            return        
        
        for roi_id, pts_list in roi_hole_points_dict.items():
            if not pts_list:
                continue

            pts_cat = np.concatenate(pts_list, axis=0)
            if pts_cat.size == 0:
                continue

            center = pts_cat.mean(axis=0).astype(np.float64)

            dist = np.linalg.norm(base_pts - center[None, :], axis=1)
            crop_idx = np.where(dist <= float(pad))[0]
            if crop_idx.size == 0:
                print(f"[ROI {roi_id}] crop empty (pad={pad})")
                continue

            pcd_crop = pcd_base.select_by_index(crop_idx.tolist())
            pcd_filtered, inlier_idx = pcd_crop.remove_radius_outlier(
                nb_points=30,
                radius=1.0
            )

            pcd_near, pcd_far, plane = self.filter_points_near_plane(pcd_filtered, distance_threshold=0.05)
            pcd_near.paint_uniform_color((0,1,0))
            pcd_far.paint_uniform_color((0.6,0.6,0.6))
            o3d.visualization.draw_geometries([pcd_near, pcd_far])
            # s = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
            # s.translate(center)
            # s.paint_uniform_color((1.0, 0.0, 0.0))  # center = red
            
            # pcd_hole = o3d.geometry.PointCloud()
            # pcd_hole.points = o3d.utility.Vector3dVector(pts_cat.astype(np.float64))
            # pcd_hole.paint_uniform_color((0.0, 0.0, 1.0))

            # o3d.visualization.draw_geometries([pcd_filtered, pcd_hole, s],window_name=f"ROI {roi_id} crop (pad={pad})")

    def filter_points_near_plane(self, pcd: o3d.geometry.PointCloud,
                             distance_threshold: float = 1.0,
                             ransac_thresh: float = 1.0,
                             ransac_n: int = 3,
                             num_iterations: int = 2000):
        pts = np.asarray(pcd.points, dtype=np.float64)
        if pts.size == 0:
            return None, None, np.array([], dtype=np.int64)

        # 1) plane fit: ax + by + cz + d = 0
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

        # 2) point-to-plane distance
        # dist = |a x + b y + c z + d| / sqrt(a^2+b^2+c^2)
        signed = (pts @ n + d) / n_norm

        keep = signed > float(distance_threshold)
        keep_idx = np.where(keep)[0]

        pcd_kept = pcd.select_by_index(keep_idx.tolist())
        pcd_removed = pcd.select_by_index(keep_idx.tolist(), invert=True)
        return pcd_kept, pcd_removed, plane_model

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":    
    main()