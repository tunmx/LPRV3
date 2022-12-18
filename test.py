import breezelpr as bpr
import cv2

# image = cv2.imread("/Users/tunm/datasets/oinbagCrawler_vertex_rec_r2/val/crop_imgs/浙BD61833.jpg")
#
# infer = bpr.PPRCNNRecognitionMNN("resource/rec/rec_ptocr_v3_rec_infer.mnn", "resource/rec/ppocr_keys_v1.txt",
#                          input_size=(48, 320))
#
#
# result = infer(image)
#
# print(result)

image = cv2.imread("/Users/tunm/datasets/oinbagCrawler/classify/green/_1_苏AD83778.jpg")

det = bpr.Y5rkDetectorMNN("resource/det/y5s_r_det_320x.mnn", input_size=(320, 320))

boxes, classes, scores = det(image)

box = boxes[0]