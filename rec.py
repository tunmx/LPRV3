import cv2

from hyperlpr3.inference.recognition import PPRCNNRecognitionORT

net16 = PPRCNNRecognitionORT("resource/rec/plate_rec_ptocr_v3_rec_infer_160.onnx", input_size=(48, 160))
net32 = PPRCNNRecognitionORT("resource/rec/rec_ptocr_v3_rec_infer.onnx", input_size=(48, 320))

align = cv2.imread("/Users/tunm/work/ZephyrLPR/resource/images/align.jpg")

print(net16(align))
print(net32(align))