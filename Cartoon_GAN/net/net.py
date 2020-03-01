# @Time    : 2020/2/29 15:18
# @Author  : FRY--
# @FileName: net.py
# @Software: PyCharm
# @Blog    ：https://fryddup.github.io

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
'''
 nn.ConvTranspose2d(  # stride(input_w-1)+k-2*Padding
                in_channels=100,
                out_channels=64 * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
'''

class Generator(nn.Module):
# '''
# 5 deconv layers
# '''
    def __init__(self):
        super(Generator, self).__init__()

        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.deconv_4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.deconv_5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=5, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Tanh(), # -1 - 1 0 mid
        )

    def forward(self, x):
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.deconv_4(x)
        x = self.deconv_5(x)

        return  x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # batchsize,3,96,96
                in_channels=3,
                out_channels=64,
                kernel_size=5,
                padding=1,
                stride=3,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(.2, inplace=True),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False, ),  # batchsize,16,32,32
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(.2, inplace=True),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(.2, inplace=True),

        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(.2, inplace=True),

        )
        self.output = nn.Sequential(
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  #
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.output(x)
        return x

#
# if __name__ == "__main__":
#     g = Generator().cuda()
#     d = Discriminator().cuda()
#     input = torch.randn(100, 100, 1, 1).cuda()
#     a = g(input)
#     b = d(a) # [100,1,1,1]
#     print("done")