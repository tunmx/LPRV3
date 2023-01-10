from breezelpr.recognition import PPRCNNRecognitionDNN
import cv2

rec = PPRCNNRecognitionDNN("resource/rec/rec_ptocr_v3_rec_infer.onnx", "resource/rec/ppocr_keys_v1.txt",
                           input_size=(48, 320))

pad = cv2.imread("pad.jpg")

out = rec(pad)

print(out)