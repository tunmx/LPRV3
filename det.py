import cv2

from hyperlpr3.inference.multitask_detect import MultiTaskDetectorMNN
from hyperlpr3.inference.multitask_detect import MultiTaskDetectorORT


mnn_net = MultiTaskDetectorMNN("resource/det/y5fu_320x_sim.mnn", input_size=(320, 320))
ort_net = MultiTaskDetectorORT("resource/det/y5fu_320x_sim.onnx", input_size=(320, 320))

img = cv2.imread("/Users/tunm/Downloads/aaaa.jpg")
# img = cv2.cvtColor(img)

m_out = mnn_net(img)
o_out = ort_net(img)

print(m_out)
print("-"*100)
print(o_out)