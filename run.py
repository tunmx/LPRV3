# -*- coding: utf-8 -*-
import hyperlpr3 as lpr3
import cv2
import urllib
import numpy as np
import re

type_list = ["蓝牌", "黄牌单层", "白牌单层", "绿牌新能源", "黑牌港澳", "香港单层", "香港双层", "澳门单层", "澳门双层", "黄牌双层"]

def is_http_url(s):
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    if regex.match(s):
        return True
    else:
        return False


def url_to_image(url):
    print(is_http_url(url))
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image



catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_HIGH)

img = url_to_image("https://tunm.oss-cn-hangzhou.aliyuncs.com/hyperlpr3/test_folder/plate_test.png")

# img = cv2.imread("/Users/tunm/datasets/boundingbox/[[277, 1572], [379, 1572], [379, 1597], [277, 1597]].jpg")

result = catcher(img)

for res in result:
    print(res)




