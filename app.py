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

class Simple3DView(QOpenGLWidget):    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)  # 필요 시 조정

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Body Hole Auto Insepction System")
        self.resize(1920, 1020)
        root = QWidget()
        self.setCentralWidget(root)
        
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

        rightLayout.addWidget(QLabel("Source Data Path"))
        self.pathEdit = QLineEdit()
        self.pathEdit.setPlaceholderText("Enter source data floder path")
        rightLayout.addWidget(self.pathEdit)

        self.btnLoad = QPushButton("Load Source Data")
        self.btnMerge = QPushButton("Merge")
        self.btnInspect = QPushButton("Inspect")

        rightLayout.addWidget(self.btnLoad)
        rightLayout.addWidget(self.btnMerge)
        rightLayout.addWidget(self.btnInspect)
        rightLayout.addWidget(QLabel("Log"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        rightLayout.addWidget(self.log, 1)

        self.btnLoad.clicked.connect(self.on_load_folder)
        self.btnMerge.clicked.connect(self.on_merge)
        self.btnInspect.clicked.connect(self.on_inspect)

    def on_load_folder(self):
        path = self.pathEdit.text().strip()
        self.log.append(f"Loading source data from: {self.current_model()}")

    def on_merge(self):
        self.log.append("Merging data...")

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