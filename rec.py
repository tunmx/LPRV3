import cv2

from hyperlpr3.inference.recognition import PPRCNNRecognitionORT

net = PPRCNNRecognitionORT("resource/rec/rec_ptocr_v3_rec_infer_160.onnx", input_size=(48, 160))

align = cv2.imread("/Users/tunm/work/ZephyrLPR/resource/images/align.jpg")

print(net(align))