import cv2

from hyperlpr3.inference.multitask_detect import MultiTaskDetectorMNN
from hyperlpr3.inference.multitask_detect import MultiTaskDetectorORT


# mnn_net = MultiTaskDetectorMNN("resource/det/y5fu_320x_sim.mnn", input_size=(320, 320))
ort_net = MultiTaskDetectorORT("resource/det/y5fu_640x_sim.onnx", input_size=(640, 640))

img = cv2.imread("/Users/tunm/datasets/boundingbox/[[360, 765], [463, 765], [463, 789], [360, 789]].jpg")
# img = cv2.cvtColor(img)

# m_out = mnn_net(img)
o_out = ort_net(img)

# for res in m_out:
for res in o_out:
    x1, y1, x2, y2 = res[:4].astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# cv2.imwrite("best.jpg", img)
cv2.imshow("q", img)
cv2.waitKey(0)

# print(m_out)
print("-"*100)
# print(m_out)