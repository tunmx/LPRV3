import cv2
import breezelpr as bpr
import click
import numpy as np

CLASSES = ['plate', ]



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
    detector = bpr.DetectorOrt(det_onnx, box_threshold=0.4)
    vertex = bpr.VertexOrt(vertex_onnx, )
    image = cv2.imread(image)
    boxes, classes, scores = detector(image)
    if boxes:
        for box in boxes:
            warped, p, mat = bpr.align_box(image, box, scale_factor=1.2, size=96)
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
