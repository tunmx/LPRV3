from breezelpr.recognition import PPRCNNRecognitionDNN
from breezelpr.recognition import PPRCNNRecognitionORT
import cv2

rec_dnn = PPRCNNRecognitionDNN("resource/rec/rec_ptocr_v3_rec_infer.onnx", "resource/rec/ppocr_keys_v1.txt",
                               input_size=(48, 320))

rec_ort = PPRCNNRecognitionORT("resource/rec/rec_ptocr_v3_rec_infer.onnx", "resource/rec/ppocr_keys_v1.txt",
                               input_size=(48, 320))

pad = cv2.imread("pad.jpg")

out_dnn = rec_dnn(pad)
out_ort = rec_ort(pad)

print('dnn result', out_dnn)
print('ort result', out_ort)