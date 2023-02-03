import cv2

from hyperlpr3.inference.recognition import PPRCNNRecognitionORT

net16 = PPRCNNRecognitionORT("resource/rec/ch_ptocr_v2_rec_infer_mydict.onnx", input_size=(32, 320))
# net32 = PPRCNNRecognitionORT("resource/rec/rec_ptocr_v3_rec_infer.onnx", input_size=(48, 320))

align = cv2.imread("sb.jpg")

print(net16(align))
# print(net32(align))