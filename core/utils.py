from enum import Enum, auto
import json
from pathlib import Path
from tqdm import tqdm
import imageio.v2 as iio
import numpy as np
import cv2

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

    def create_mask_from_image_file(self, z_with_path, z_without_path, mask_save_path, tolerance=1, min_area = 200, use_morph = True) :
        # 1. 로드
        z_with = iio.imread(z_with_path).astype(np.float32)
        z_without = iio.imread(z_without_path).astype(np.float32)

        assert z_with.shape == z_without.shape, "z.tif 크기가 다릅니다"

        # 2. invalid depth 제거 (중요)
        valid = (z_with > 0) & (z_without > 0) & np.isfinite(z_with) & np.isfinite(z_without)

        # 3. smoothing (바닥 노이즈 제거 핵심)
        z_with_s = cv2.medianBlur(z_with, 5)
        z_without_s = cv2.medianBlur(z_without, 5)

        # 4. diff 계산
        diff = np.abs(z_with_s - z_without_s)

        # 5. mask 생성
        mask = np.zeros_like(diff, dtype=np.uint8)
        mask[(diff > tolerance) & valid] = 255

        # 5-1) morphology로 점노이즈 완화(선택)
        if use_morph:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)   # 작은 점 제거
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)  # 작은 구멍 메움
       
        mask = self.polish_mask(mask, close_ksize=11, close_iter=2, min_area=min_area)

        # 6. 저장
        cv2.imwrite(mask_save_path, mask)
        return mask       
    
    def create_mask_from_depth_array(self, z_with, z_without, mask_save_path, tolerance=1, min_area=200, use_morph=True):
        # 1. 로드 및 타입 변환
        # 만약 z_with가 이미 배열이라면 astype만 수행, z_without이 경로라면 로드
        z_with = z_with.astype(np.float32)
        
        if isinstance(z_without, str):
            z_without = iio.imread(z_without).astype(np.float32)
        else:
            z_without = z_without.astype(np.float32)

        assert z_with.shape == z_without.shape, "Depth 데이터의 크기가 서로 다릅니다."

        # 2. invalid depth 제거 (0 이하 또는 무한대 값 제외)
        valid = (z_with > 0) & (z_without > 0) & np.isfinite(z_with) & np.isfinite(z_without)

        # 3. smoothing (바닥 노이즈 제거 핵심)
        # medianBlur는 float32를 직접 지원하지 않는 경우가 있으므로 필요한 경우 uint16 변환 후 처리하거나 그대로 시도
        z_with_s = cv2.medianBlur(z_with, 5)
        z_without_s = cv2.medianBlur(z_without, 5)

        # 4. diff 계산
        diff = np.abs(z_with_s - z_without_s)

        # 5. mask 생성
        mask = np.zeros_like(diff, dtype=np.uint8)
        mask[(diff > tolerance) & valid] = 255

        # 5-1) morphology로 점노이즈 완화
        if use_morph:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)   # 작은 점 제거
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)  # 작은 구멍 메움
        
        # 기존에 정의하신 polish_mask 호출
        mask = self.polish_mask(mask, close_ksize=11, close_iter=2, min_area=min_area)

        # 6. 저장
        cv2.imwrite(mask_save_path, mask)
        
        return mask
    
    def remove_small_components(self, mask_u8: np.ndarray, min_area: int = 200) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

        cleaned = np.zeros_like(mask_u8)
        # stats: [label] -> (x, y, w, h, area)
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == lbl] = 255
        return cleaned
    
    def polish_mask(self, mask_u8: np.ndarray, close_ksize: int = 9, close_iter: int = 2, open_ksize: int = 3, open_iter: int = 1, min_area: int = 200) -> np.ndarray:
        mask = (mask_u8 > 0).astype(np.uint8) * 255

        # 1) 작은 점 제거 (원하면 먼저 open)
        if open_ksize > 1:
            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=open_iter)

        # 2) 구멍/틈 메우기 (close 강하게)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=close_iter)

        # 3) 작은 군집 제거
        mask = self.remove_small_components(mask, min_area=min_area)

        # 4) 내부 hole fill (안전 버전)
        mask = self.fill_holes_only(mask)

        return mask
    
    def fill_holes_only(self, mask_u8: np.ndarray) -> np.ndarray:
        """
        mask_u8: 0/255 바이너리 마스크
        반환: 마스크 내부의 hole만 채워진 0/255 마스크
        """
        # 1) 0/255 보장
        mask = (mask_u8 > 0).astype(np.uint8) * 255

        # 2) 테두리 접촉 문제 방지: 0으로 테두리 한 겹 추가
        mask_pad = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

        h, w = mask_pad.shape

        # 3) 배경(0)에서 flood fill로 바깥 영역만 채우기
        flood = mask_pad.copy()
        ffmask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, ffmask, seedPoint=(0, 0), newVal=255)

        # flood는 "바깥 배경"이 255로 채워진 상태.
        # 4) flood를 반전하면, 바깥은 0, "안쪽 hole"만 255로 남는다.
        holes = cv2.bitwise_not(flood)

        # 5) 원래 mask와 hole을 OR → hole만 채워진 결과
        filled = cv2.bitwise_or(mask_pad, holes)

        # 6) padding 제거
        filled = filled[1:-1, 1:-1]

        return filled