# @Time    : 2020/2/29 17:25
# @Author  : FRY--
# @FileName: data_loader.py
# @Software: PyCharm
# @Blog    ：https://fryddup.github.io

import os
import cv2

path = 'E:/Cartoon_Gan/data/faces/'
images = []

def get_img():
    for img in os.listdir(path):
        img_data = cv2.imread(path + img)
        images.append(img_data)
    return images
#
# if __name__ == "__main__":
#     a = get_img()
#     print("done")