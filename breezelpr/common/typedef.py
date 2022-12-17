import numpy as np


class Plate(object):

    def __init__(self,
                 vertex: np.ndarray,
                 plate_code: str,
                 rec_confidence: float,
                 det_bound_box,
                 dex_bound_confidence: float, ):
        assert vertex.shape == (4, 2)
        self.vertex = vertex
        self.det_bound_box = det_bound_box
        self.plate_code = plate_code
        self.rec_confidence = rec_confidence
        self.dex_bound_confidence = dex_bound_confidence

        self.left_top, self.right_top, self.right_bottom, self.left_bottom = vertex

    def to_dict(self):
        return dict(plate_code=self.plate_code, rec_confidence=self.rec_confidence, vertex=self.vertex,
                    det_bound_box=self.det_bound_box, dex_bound_confidence=self.dex_bound_confidence)

    def __dict__(self):
        return self.to_dict()

    def __str__(self):
        return str(self.to_dict())
