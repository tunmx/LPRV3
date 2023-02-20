import cv2

from hyperlpr3.inference.recognition import PPRCNNRecognitionORT

# def read_key_file(path: str) -> list:
#     key_map = ['blank', ]
#     try:
#         with open(path, 'r') as f:
#             lines = f.readlines()
#             key_map += [item.strip() for item in lines]
#     except Exception as err:
#         print(f"读取key文件发生错误: {err}")
#
#     return key_map
#
#
# token = read_key_file("resource/rec/plate.txt")
# print(token)

net16 = PPRCNNRecognitionORT("resource/rec/rec_mdice_160_ptocr_v3_rec_infer_160.onnx", input_size=(48, 160))
# net32 = PPRCNNRecognitionORT("resource/rec/rec_ptocr_v3_rec_infer.onnx", input_size=(48, 320))

align = cv2.imread("resource/images/sb48x160.jpg")

print(net16(align))
# print(net32(align))