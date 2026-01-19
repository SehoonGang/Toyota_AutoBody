from enum import Enum, auto
import json
from pathlib import Path
from tqdm import tqdm

class FileType(Enum) : 
    Image = auto()
    Calibration = auto()
    DeepLearningModel = auto()    

class Utils :
    def __init__(self):
        self._calibration_file_path = ""
        self._source_data_folder_files = []
        self._deep_leaning_model_file_path = ""
        self._cad_data = []
        self._load_cad_data()

    @property
    def calibration_file_path(self) -> str:
        return self._calibration_file_path
    @property
    def source_data_folder_files(self) -> list[tuple[str, str, str, str, str, str]]:
        return self._source_data_folder_files
    @property
    def deep_leaning_model_file_path(self) -> str:
        return self._deep_leaning_model_file_path    
    @property
    def cad_data(self) -> any:
        return self._cad_data

    def on_load_source_data_folder(self, path: str, file_type: FileType) :
        if (file_type == FileType.Image) :
            root = Path(path)
            files = [p for p in root.rglob("*.png") if p.is_file() and "kungfu" not in str(p).lower()]
            files = [str(p) for p in sorted(files, key=lambda p: p.name)]            
            source_data = []

            for path in tqdm(files, desc="Loading Source Data") :
                source_data.append([
                    path,
                    path.replace("Texture_8Bit.png", "PointCloud_X.tif"),
                    path.replace("Texture_8Bit.png", "PointCloud_Y.tif"),
                    path.replace("Texture_8Bit.png", "PointCloud_Z.tif"),
                    path.replace("IMG_Texture_8Bit.png", "SCAN_POSE.txt"),
                    path.replace("IMG_Texture_8Bit.png", "Mask.tiff"),
                ])
            self._source_data_folder_files= source_data

        elif (file_type == FileType.Calibration) :
            self._calibration_file_path = path

        else :
            self._deep_leaning_model_file_path = path
    def _load_cad_data(self):
        json_path = r".\\data\\cad.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self._cad_data = data