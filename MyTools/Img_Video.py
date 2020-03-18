# ------------------------------------------------------------------------------
# # @Time    : 2020/3/18 下午 6:11
# # @Author  : fry
# @FileName: Img_Video.py
# @Functions:
# ①. Split(Video to image) Compose(images sequence to video)
# ②. Crop image
# ------------------------------------------------------------------------------

import cv2
import numpy as np
import os
import tqdm
from .heatMap_to_coordinates import get_affine_transform

# 视 频 分 解 合 成
class VideoData(object):
    def __init__(self, path):

        self.path = path # E:\Research\Code\Improved-Body-Parts\demo\aaa\video.mp4
        self.file_path, self.full_file_name = os.path.split(self.path) # E:\Research\Code\Improved-Body-Parts\demo\aaa\
        self.file_name,self.extra_name = os.path.splitext(self.full_file_name)
        self.cap = cv2.VideoCapture(self.path)
        assert self.cap.isOpened(), 'Cannot capture source'
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.datalen = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Video Split
    def VideoSplit(self):
        self.get_Info()
        assert self.cap.isOpened(), "Can't find video"
        for i in range(self.datalen):
            (flag, frame) = self.cap.read()
            filename = self.file_path + "/" + self.file_name + "_" + str(i) + '.jpg'
            if flag == True:
                cv2.imwrite(filename, frame)
                # img.append(frame)
        print("视频拆分单帧完成！")

    # img Compose
    def ImageCompose(self):
        self.get_Info()
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_path = self.file_path+"/"+"video_out" #+"/"'Compose.mp4'
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        write_name = out_path + "/Compose_30f.mp4"
        videoWrite = cv2.VideoWriter(write_name, fourcc, 30, (self.width, self.height))  # 写入对象：1.fileName  2.-1：表示选择合适的编码器  3.视频的帧率  4.视频的size
        for image in tqdm(os.listdir(self.file_path)):
            fileName = self.file_path + "/" + image
            img = cv2.imread(fileName)
            videoWrite.write(img)  # 写入方法  1.编码之前的图片数据
        print('视频合成完成')

    # 返回视频信息
    def get_Info(self):
        print ("Fps: %f, Width: %d, Height: %d, Length: %d"%(self.fps, self.width, self.height, self.datalen))
        return

    def length(self):
        return self.datalen

# 图像裁剪
def Img_crop(img, box, size):
    '''

    :param img: path
    :param box: [left, top, width, height]
    :param size: [target_width, target_height]
    :return:
    '''
    data_numpy = cv2.imread(img, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if data_numpy is None:
        raise ValueError('Fail to read {}'.format(img))
    c, s = _box2cs(box, data_numpy.shape[0], data_numpy.shape[1])
    r = 0
    trans = get_affine_transform(c, s, r, size)
    input = cv2.warpAffine(
        data_numpy,
        trans,
        (size[0], size[1]),
        flags=cv2.INTER_LINEAR)

    # write file
    cv2.imwrite('/a.jpg',input)


def _box2cs(box, image_width, image_height):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, image_width, image_height)


def _xywh2cs(x, y, w, h, image_width, image_height):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    aspect_ratio = image_width * 1.0 / image_height
    pixel_std = 200

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


if __name__ == "__main__":
    video = VideoData('E:/Research/Code/Improved-Body-Parts/demo/video/output/c.mp4')
    video.ImageCompose()