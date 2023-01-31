# -*- coding: utf-8 -*-
import hyperlpr3 as lpr3
import cv2
import urllib
import numpy as np


def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

catcher = lpr3.LicensePlateCatcher()

# img = url_to_image("https://tunm.oss-cn-hangzhou.aliyuncs.com/hyperlpr3/test_folder/plate_test.png")

img = cv2.imread("/Users/tunm/Downloads/bug.jpg")

result = catcher(img)
print(result)