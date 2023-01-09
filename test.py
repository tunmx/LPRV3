import breezelpr as bpr
import cv2
from breezelpr.multitask_detect import MultiTaskDetectorORT
from breezelpr.multitask_detect import MultiTaskDetectorMNN
from breezelpr.multitask_detect import MultiTaskDetectorDNN

# image = cv2.imread("/Users/tunm/datasets/oinbagCrawler_vertex_rec_r2/val/crop_imgs/浙BD61833.jpg")
#
# infer = bpr.PPRCNNRecognitionMNN("resource/rec/rec_ptocr_v3_rec_infer.mnn", "resource/rec/ppocr_keys_v1.txt",
#                          input_size=(48, 320))
#
#
# result = infer(image)
#
# print(result)

# image = cv2.imread("/Users/tunm/datasets/plate_dataset_various/低速车-11/鲁R06168.jpg")

# det = bpr.Y5rkDetectorMNN("resource/det/y5s_r_det_320x.mnn", input_size=(320, 320))
#
# boxes, classes, scores = det(image)
#
# box = boxes[0]

# det = MultiTaskDetectorMNN("/Users/tunm/work/Chinese_license_plate_detection_recognition/onnx/y5fu_320x_sim.mnn",
#                            input_size=(320, 320))
# outputs = det(image)
# print(outputs)
#
# for item in outputs:
#     rect = item[:4].astype(int)
#     score = item[4]
#     land_marks = item[5:13].reshape(4, 2).astype(int)
#     classify = item[13]
#     print(f"type: {classify}")
#     x1, y1, x2, y2 = rect
#     cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 200), 1)
#     for x, y in land_marks:
#         cv2.line(image, (x, y), (x, y), (0, 0, 255), 3)
#         cv2.imshow("img", image)
#         cv2.waitKey(0)

#
# cv2.imshow("img", image)
# cv2.waitKey(0)


image = cv2.imread("/Users/tunm/Downloads/319931671972510_.pic_hd.png")

det = MultiTaskDetectorDNN("/Users/tunm/work/Chinese_license_plate_detection_recognition/onnx/y5fu_320x_sim.onnx",
                           input_size=(320, 320))

det(image)
