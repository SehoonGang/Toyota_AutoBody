class Detect:
    def __init__(self):
        return

    def find_circle_center(self, texture_image) :
        detect_circle_setting = dict(topk_hough=5,
                                     roi_scale=2.2,
                                     roi_half_min=60,
                                     roi_half_max=260,
                                     band_px=4.0,
                                     arc_min=0.15,
                                     dp=1.2,
                                     minDist=140,
                                     param1=120,
                                     param2=24,
                                     minRadius=50,
                                     maxRadius=60)
        