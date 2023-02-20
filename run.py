# -*- coding: utf-8 -*-
import hyperlpr3 as lpr3
import cv2
import urllib
import numpy as np

type_list = ["蓝牌", "黄牌单层", "白牌单层", "绿牌新能源", "黑牌港澳", "香港单层", "香港双层", "澳门单层", "澳门双层", "黄牌双层"]

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

catcher = lpr3.LicensePlateCatcher()

# img = url_to_image("https://tunm.oss-cn-hangzhou.aliyuncs.com/hyperlpr3/test_folder/plate_test.png")

img = cv2.imread("/Users/tunm/datasets/plate_dataset_various/“蒙”牌-46/蒙A6FH93.jpg")

result = catcher(img)
print(result)

for res in result:
    print(f"{res['plate_code']}, {res['rec_confidence']}, {type_list[res['plate_type']]}")




