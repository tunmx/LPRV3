import cv2
import numpy as np

fk = np.load("fk.npy")
print(fk)

fk = fk.transpose((1, 2, 0)) * 127.5 + 127.5

img = fk.astype(np.uint8)

# cv2.imshow("w", img)
# cv2.waitKey(0)

cv2.imwrite("a.jpg", img)