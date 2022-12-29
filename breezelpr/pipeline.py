import numpy as np

from .common.typedef import Plate
from .common.tools_process import *


class LPRMultiTaskPipeline(object):

    def __init__(self, detector, recognizer):
        self.detector = detector
        self.recognizer = recognizer

    def run(self, image: np.ndarray) -> list:
        result = list()
        outputs = self.detector(image)
        for out in outputs:
            rect = out[:4].astype(int)
            score = out[4]
            land_marks = out[5:13].reshape(4, 2).astype(int)
            pad = get_rotate_crop_image(image, land_marks)
            plate_code, rec_confidence = self.recognizer(pad)
            if plate_code == '':
                continue
            plate = Plate(vertex=land_marks, plate_code=plate_code, det_bound_box=np.asarray(rect),
                          rec_confidence=rec_confidence, dex_bound_confidence=score)
            result.append(plate.to_dict())

        return result

    def __call__(self, image: np.ndarray, *args, **kwargs):
        return self.run(image)


class LPRPipeline(object):

    def __init__(self, detector, vertex_predictor, recognizer, ):
        self.detector = detector
        self.vertex_predictor = vertex_predictor
        self.recognizer = recognizer

    # @cost("PipelineTotalCost")
    def run(self, image: np.ndarray) -> list:
        result = list()
        boxes, classes, scores = self.detector(image)
        fp_boxes_index = find_the_adjacent_boxes(boxes)
        print('检测到挨近框:', fp_boxes_index)
        image_blacks = list()
        if len(fp_boxes_index) > 0:
            for idx in fp_boxes_index:
                image_black = np.zeros_like(image)
                box = boxes[idx]
                x1, y1, x2, y2 = np.asarray(box).astype(int)
                image_black[y1:y2, x1:x2] = image[y1:y2, x1:x2]
                image_blacks.append(image_black)
        if boxes:
            fp = 0
            for idx, box in enumerate(boxes):
                det_confidence = scores[idx]
                if idx in fp_boxes_index:
                    warped, p, mat = align_box(image_blacks[fp], box, scale_factor=1.2, size=96)
                    fp += 1
                else:
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
                # cv2.imshow("pad", pad)
                # cv2.waitKey(0)
                plate_code, rec_confidence = self.recognizer(pad)
                if plate_code == '':
                    continue
                plate = Plate(vertex=trans_points, plate_code=plate_code, det_bound_box=np.asarray(box),
                              rec_confidence=rec_confidence, dex_bound_confidence=det_confidence)
                result.append(plate.to_dict())

        return result

    def __call__(self, image: np.ndarray, *args, **kwargs):
        return self.run(image)
