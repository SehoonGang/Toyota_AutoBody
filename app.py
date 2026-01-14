import json
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
from core import Utils, FileType, PCD
import open3d as o3d
import numpy as np
from tqdm import tqdm
import imageio.v2 as iio
import cv2
from ultralytics import YOLO

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
        self.radioR.setChecked(True)
        self.radioGroup.addButton(self.radioL)
        self.radioGroup.addButton(self.radioR) 
        radioRow = QHBoxLayout()
        radioRow.addWidget(self.radioL)
        radioRow.addWidget(self.radioR)
        radioRow.addStretch(1)
        rightLayout.addLayout(radioRow)

        sourceDataFolderRow = QHBoxLayout()
        sourceDataFolderRow.addWidget(QLabel("Source Data"))
        self.tbSourceDataFolderPath = QLineEdit(rf"C:\Users\SehoonKang\Desktop\dataset\260113_Scan\260113_Scan\RH")
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
        self.tbDeepLearningModelFilePath = QLineEdit(rf"C:\Users\SehoonKang\Desktop\dataset\260113_Scan\260113_Scan\best.pt")        
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
        self.log.append("Start to merge frames")
        T_list, merged_pcd, circle_points_merged = self.pcd.merge_pcd(self.utils.source_data_folder_files, self.utils.calibration_file_path, "fanuc", self.current_model())
        self.T_list = T_list

        T_array = np.stack(T_list, axis=0)
        np.save(rf"C:\Users\SehoonKang\Desktop\dataset\260113_Scan\260113_Scan\body.npy", T_array)

        #변경 필요 ========================================
        json_path = r".\\data\\cad.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cad_centers_array = np.array(data["LH"]["cad_centers"], dtype=np.float32)
        #=================================================

        # moved_merge_pcd, T_to_cad, report = self.pcd.move_merged_pcd_to_cad(merged_pcd=merged_pcd,
        #                                                                     CAD_CENTERS=cad_centers_array,
        #                                                                     align_points=np.asarray(circle_points_merged, dtype=np.float64),
        #                                                                     copy_pcd=True)
        
        # self.result_pcd = moved_merge_pcd
        # self.result_T = T_to_cad

        pcd = merged_pcd.voxel_down_sample(1.0)
        self.set_pointcloud(pcd)
        self.log.append("merge frames complete.")

    def on_inspect(self):
        self.log.append("Inspecting data...")
        roi_hole_points_dict = {}

        for i, frame in enumerate(tqdm(self.utils.source_data_folder_files, total=len(self.utils.source_data_folder_files))):
            texture_path, x_path, y_path, z_path, pose_path, mask_path = frame

            X = iio.imread(x_path).astype(np.float64)
            Y = iio.imread(y_path).astype(np.float64)
            Z = iio.imread(z_path).astype(np.float64)

            h, w = X.shape
            X = np.asarray(X)
            Y = np.asarray(Y)
            Z = np.asarray(Z)

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
            pts_cad = self.transform_points(pts_cam, T_cam_to_cad)

            image_for_seg = cv2.imread(texture_path, cv2.IMREAD_COLOR)
            img_h, img_w, _ = image_for_seg.shape

            #변경 필요 ========================================
            json_path = r".\\data\\cad.json"
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cad_points  = np.array(data["RH"]["cad_welding_points"], dtype=np.float32)            

            for roi_id, center in enumerate(cad_points, start=1):
                dist = np.linalg.norm(pts_cad - center, axis=1)
                mask_roi_3d = dist <= 4
                num_roi_pts = np.count_nonzero(mask_roi_3d)

                if num_roi_pts == 0:
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
                    continue

                crop_img = image_for_seg[y_min:y_max + 1, x_min:x_max + 1]
                if crop_img.size == 0:
                    continue
                
                ch, cw, _ = crop_img.shape
                if ch < 16 or cw < 16:                
                    continue

                results = self.seg_model(crop_img, verbose=False)
                if len(results) == 0 or results[0].masks is None:
                    print(f"[INFO] ROI : YOLO mask 없음 (view {i}).")
                    continue

                masks_yolo = results[0].masks.data.cpu().numpy()  # (K, Hc, Wc)
                if masks_yolo.shape[0] == 0:
                    print(f"[INFO] ROI : mask 개수 0 (view {i}).")
                    continue

                mask_bin = (masks_yolo > 0.5)
                full_mask_local = np.any(mask_bin, axis=0)  # (Hm, Wm) bool

                mask_resized = cv2.resize(
                    full_mask_local.astype(np.uint8),
                    (cw, ch),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

                mask_full = np.zeros((img_h, img_w), dtype=bool)
                mask_full[y_min:y_max + 1, x_min:x_max + 1] = mask_resized

                mask_on_pixels = mask_full[ys_idx, xs_idx]  # (N,) bool
                mask_hole = mask_on_pixels & mask_roi_3d

                roi_hole_pts = pts_cad[mask_hole]
                n_hole = roi_hole_pts.shape[0]

                if n_hole > 0:                
                    roi_hole_points_dict.setdefault(roi_id, []).append(roi_hole_pts)


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

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":    
    main()