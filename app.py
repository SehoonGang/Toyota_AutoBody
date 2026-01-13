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
from core import Utils, FileType, PCD
import open3d as o3d

class Simple3DView(QOpenGLWidget):    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)

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
        self.view3d = Simple3DView()
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
        self.tbSourceDataFolderPath = QLineEdit(rf"C:\Users\SehoonKang\Desktop\dataset\260103\data\withobject_last")
        sourceDataFolderRow.addWidget(self.tbSourceDataFolderPath)
        self.btnSourceDataLoad = QPushButton("Load")
        sourceDataFolderRow.addWidget(self.btnSourceDataLoad)
        rightLayout.addLayout(sourceDataFolderRow)
        
        calibrationFileRow = QHBoxLayout()
        calibrationFileRow.addWidget(QLabel("Calibration File"))
        self.tbCalibrationFilePath = QLineEdit(rf"C:\Users\SehoonKang\Desktop\dataset\260103\cam_robot_extrinsic_0_1_hand_eye.yml")
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
        self.log.append("Merging data...")
        pcd = self.pcd.merge_pcd(self.utils.source_data_folder_files, self.utils.calibration_file_path, "fanuc")
        o3d.visualization.draw_geometries([pcd])

    def on_inspect(self):
        self.log.append("Inspecting data...")

    def current_model(self):
        return 'L' if self.radioL.isChecked() else 'R'

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":    
    main()