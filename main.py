import cv2
from detect import DetectorOrt
import click

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
@click.option("-onnx", "--onnx", default="resource/det/y5s_r_det_640x.onnx",
              type=click.Path(exists=True))
@click.option("-image", "--image", type=click.Path(exists=True))
def run(onnx, image):
    detector = DetectorOrt(onnx)
    image = cv2.imread(image)
    boxes, classes, scores = detector(image)
    if boxes:
        draw(image, boxes, scores, classes)
    cv2.imshow("post process result", image)
    cv2.waitKeyEx(0)

    print(boxes)


if __name__ == '__main__':
    run()
