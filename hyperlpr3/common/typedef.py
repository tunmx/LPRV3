import numpy as np

PLATE_TYPE_BLUE = 0
PLATE_TYPE_GREEN = 1
PLATE_TYPE_YELLOW = 2

INFER_ONNX_RUNTIME = 0
INFER_MNN = 1

DETECT_LEVEL_LOW = 0
DETECT_LEVEL_HIGH = 1

class Plate(object):

    def __init__(self,
                 vertex: np.ndarray,
                 plate_code: str,
                 rec_confidence: float,
                 det_bound_box,
                 dex_bound_confidence: float,
                 plate_type: int):
        assert vertex.shape == (4, 2)
        self.vertex = vertex
        self.det_bound_box = det_bound_box
        self.plate_code = plate_code
        self.rec_confidence = rec_confidence
        self.dex_bound_confidence = dex_bound_confidence

        self.left_top, self.right_top, self.right_bottom, self.left_bottom = vertex
        self.plate_type = plate_type

    def to_dict(self):
        return dict(plate_code=self.plate_code, rec_confidence=self.rec_confidence,
                    det_bound_box=self.det_bound_box, plate_type=self.plate_type)

    def __dict__(self):
        return self.to_dict()

    def __str__(self):
        return str(self.to_dict())
