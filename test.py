import breezelpr as bpr
import cv2

image = cv2.imread("/Users/tunm/datasets/oinbagCrawler_vertex_rec_r2/val/crop_imgs/浙BD61833.jpg")

infer = bpr.PPRCNNRecognitionMNN("resource/rec/rec_ptocr_v3_rec_infer.mnn", "resource/rec/ppocr_keys_v1.txt",
                         input_size=(48, 320))


result = infer(image)

print(result)