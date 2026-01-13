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

class PointCloudView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=2000)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)

        self.scatter = None

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
        self.tbSourceDataFolderPath = QLineEdit(rf"C:\Users\gsh72\toyota-auto-body\Data\ScanData\withobject\withobject_last")
        sourceDataFolderRow.addWidget(self.tbSourceDataFolderPath)
        self.btnSourceDataLoad = QPushButton("Load")
        sourceDataFolderRow.addWidget(self.btnSourceDataLoad)
        rightLayout.addLayout(sourceDataFolderRow)
        
        calibrationFileRow = QHBoxLayout()
        calibrationFileRow.addWidget(QLabel("Calibration File"))
        self.tbCalibrationFilePath = QLineEdit(rf"C:\Users\gsh72\toyota-auto-body\Data\cam_robot_extrinsic_0_1_hand_eye.yml")
        calibrationFileRow.addWidget(self.tbCalibrationFilePath)
        self.btnCalibrationFilePath = QPushButton("Load")
        calibrationFileRow.addWidget(self.btnCalibrationFilePath)
        rightLayout.addLayout(calibrationFileRow)

        deepLearningFileRow = QHBoxLayout()
        deepLearningFileRow.addWidget(QLabel("Deep Learning"))
        self.tbDeepLearningModelFilePath = QLineEdit()        
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
        self.utils.on_load_source_data_folder(self, self.tbDeepLearningModelFilePath.text(), FileType.DeepLearningModel)
        self.log.append(rf"load {self.tbDeepLearningModelFilePath.text()} completed.")        

    def on_merge(self):
        self.log.append("Start to merge frames")
        T_list, merged_pcd, circle_points_merged = self.pcd.merge_pcd(self.utils.source_data_folder_files, self.utils.calibration_file_path, "fanuc")

        #변경 필요
        json_path = r".\\data\\cad.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cad_centers_array = np.array(data["RH"]["cad_centers"], dtype=np.float32)

        moved_merge_pcd, T_to_cad, report = self.pcd.move_merged_pcd_to_cad(merged_pcd=merged_pcd,
                                                                            CAD_CENTERS=cad_centers_array,
                                                                            align_points=np.asarray(circle_points_merged, dtype=np.float64),
                                                                            copy_pcd=True)

        pcd = moved_merge_pcd.voxel_down_sample(1.0)
        self.set_pointcloud(pcd)
        self.log.append("merge frames complete.")

    def on_inspect(self):
        self.log.append("Inspecting data...")

    def current_model(self):        
        return 'L' if self.radioL.isChecked() else 'R'
    
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

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":    
    main()