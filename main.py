import cv2
from detect import DetectorOrt, VertexOrt
import click
import numpy as np

CLASSES = ['plate', ]

def align_box(imgs, bbox, size=96, scale_factor=1.0, center_bias=0, borderValue=(0, 0, 0)):
    bias_x = (-1 + 2 * np.random.sample()) * center_bias
    bias_y = (-1 + 2 * np.random.sample()) * center_bias
    b_x1, b_y1, b_x2, b_y2 = bbox
    cx, cy = (b_x1 + b_x2) // 2, (b_y1 + b_y2) // 2
    w = b_x2 - b_x1
    h = b_y2 - b_y1
    cx += w * bias_x
    cy += h * bias_y

    base_r = max(w, h)
    j_x = 0
    j_y = 0
    j_r = 0
    base_r += j_r
    r = int(base_r / 2 * scale_factor)
    cy -= int(base_r * 0)
    cx += j_x
    cy += j_y
    x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r
    x3, y3 = cx - r, cy + r
    _x1, _y1, _x2, _y2, _x3, _y3 = [0, 0, size, size, 0, size]
    src = np.array([x1, y1, x2, y2, x3, y3], dtype=np.float32).reshape(3, 2)
    sv = np.asarray([[b_x1, b_y1, 1], [b_x2, b_y2, 1]])

    dst = np.array([_x1, _y1, _x2, _y2, _x3, _y3], dtype=np.float32).reshape(3, 2)
    assert src.dtype == np.float32
    assert dst.dtype == np.float32
    assert src.shape == (3, 2)
    assert dst.shape == (3, 2)
    mat = cv2.getAffineTransform(src, dst)
    p = sv.dot(mat.T).reshape(-1)
    if type(imgs) == list:
        imgs = [cv2.warpAffine(img, mat, (size, size), borderValue=borderValue) for img in imgs]
    else:
        imgs = cv2.warpAffine(imgs, mat, (size, size), borderValue=borderValue)
    return imgs, p, mat

def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.
    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), (0, 0, 255), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (190, 120, 188), 2)


@click.command()
@click.option("-det_onnx", "--det_onnx", default="resource/det/y5s_r_det_320x.onnx", type=click.Path(exists=True))
@click.option("-vertex_onnx", "--vertex_onnx", default="resource/vertex/vertex_mnet025_x96.onnx", type=click.Path(exists=True))
@click.option("-image", "--image", type=click.Path(exists=True))
def run(det_onnx, vertex_onnx, image):
    detector = DetectorOrt(det_onnx)
    vertex = VertexOrt(vertex_onnx, )
    image = cv2.imread(image)
    boxes, classes, scores = detector(image)
    if boxes:
        for box in boxes:
            warped, p, mat = align_box(image, box, scale_factor=1.2, size=96)
            kps = vertex(warped)
            polyline = list()
            for point in kps:
                polyline.append([point[0], point[1], 1])
            polyline = np.asarray(polyline)
            inv = cv2.invertAffineTransform(mat)
            trans_points = np.dot(inv, polyline.T).T
            cv2.polylines(image, [trans_points.astype(np.int32)], True, (0, 0, 200), 2, )
            for x, y in trans_points.astype(np.int32):
                cv2.line(image, (x, y), (x, y), (0, 240, 0), 3)

    cv2.imshow("post process result", image)
    cv2.waitKeyEx(0)

    # print(boxes)


if __name__ == '__main__':
    run()
