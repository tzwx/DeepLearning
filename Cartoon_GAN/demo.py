# @Time    : 2020/3/1 10:02
# @Author  : FRY--
# @FileName: demo.py
# @Software: PyCharm
# @Blog    ：https://fryddup.github.io
import torch
import torch.nn as nn
import cv2
import numpy as np
from net.net import  Generator

gen = torch.load('pkl/pkl69generator.pkl')

input = torch.randn(100, 100, 1, 1).cuda()

images = gen(input) # [100,3,96,96]
print("done")
for i in range(100):
    img = images[i].permute(1,2,0).cpu().detach().numpy()*255
    cv2.imwrite('output_69/'+'img_'+str(i)+'.jpg',img)
print("done")