from .common.typedef import Plate
from .common.tools_process import *


class LPRPipeline(object):

    def __init__(self, detector, vertex_predictor, recognizer, ):
        self.detector = detector
        self.vertex_predictor = vertex_predictor
        self.recognizer = recognizer

    @cost("PipelineTotalCost")
    def run(self, image: np.ndarray) -> list:
        result = list()
        boxes, classes, scores = self.detector(image)
        if boxes:
            for idx, box in enumerate(boxes):
                det_confidence = scores[idx]
                warped, p, mat = align_box(image, box, scale_factor=1.2, size=96)
                kps = self.vertex_predictor(warped)
                polyline = list()
                for point in kps:
                    polyline.append([point[0], point[1], 1])
                polyline = np.asarray(polyline)
                inv = cv2.invertAffineTransform(mat)
                trans_points = np.dot(inv, polyline.T).T
                pad = get_rotate_crop_image(image, trans_points)
                # print(pad.shape)
                plate_code, rec_confidence = self.recognizer(pad)
                if plate_code == '':
                    continue
                plate = Plate(vertex=trans_points, plate_code=plate_code, det_bound_box=np.asarray(box),
                              rec_confidence=rec_confidence, dex_bound_confidence=det_confidence)
                result.append(plate.to_dict())

        return result

    def __call__(self, image: np.ndarray, *args, **kwargs):
        return self.run(image)
