import hyperlpr3 as lpr3
import cv2

catcher = lpr3.LicensePlateCatcher()

img = cv2.imread("/Users/tunm/Downloads/s.png")

result = catcher(img)
print(result)