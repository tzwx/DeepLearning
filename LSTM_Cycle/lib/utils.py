# @Time    : 2020/1/12 14:06
# @Author  : FRY--
# @FileName: util.py
# @Software: PyCharm
# @Blog    ：https://fryddup.github.io

'''
*********************
图像数据处理工具函数
*********************
'''
import cv2
import numpy as np
from dataset import DataLoader
import time
import math

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def get_final_preds(batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                diff = np.array([hm[py][px+1] - hm[py][px-1],
                                 hm[py+1][px]-hm[py-1][px]])
                coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals

def heatmaps_mean(map1, map2):
    '''

    :param map1: [channel,width,height]
    :param map2: [3,45,45]
    :return:
    '''
    assert map1.shape != map2.shape , 'Get Different size between two heatmaps'
    return (map1 + map2)/2

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

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


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def crop_img(path_current, images, bbox):

    length = len(images)
    trans_martix = [[] for i in range(length)]
    for index in range(length):
        img_file = images[index]
        data_numpy = cv2.imread(img_file[0])
        box = bbox[index]
        box_length = len(box)
        if box_length > 1 and not(data_numpy is None):
            for j in range(box_length):
                c, s = _box2cs(box[j], data_numpy.shape[0], data_numpy.shape[1])

                r = 0
                trans = get_affine_transform(c, s, r, [368, 368])
                trans_martix[index].append(trans)
                input = cv2.warpAffine(
                    data_numpy,
                    trans,
                    (368, 368),
                    flags=cv2.INTER_LINEAR)
                cv2.imwrite(path_current + 'transform_%d_%d.png' % (index, j), input[:, :, :])
    return trans_martix

def key_to_heatmap(keypoints, trans):
    numer_of_people = len(keypoints)
    number_of_img = len(keypoints[0])
    key_transform = [[[] for i in range(number_of_img)] for i in range(numer_of_people)]
    for i in range(numer_of_people):
        for j in range(number_of_img):
            for k in range(15):
                key = np.array(keypoints[i][j][k][0:2])
                key_after = np.matmul(key, trans[i][j])
                key_transform[i][j].append(key_after)
    return key_transform


class Joints_data(object):

    def __init__(self, images, images_final):
        self.images = images
        self.images_final = images_final
        data_img = cv2.imread(self.images[0][0])
        self.width = 368
        self.height = 368
        self.num_joints = 15
        self.pixel_std = 200
        self.target_type = 'gaussian'
        self.heatmap_size = np.array([45, 45])
        self.sigma = 2
        self.image_size = np.array([self.width, self.height])
        self.num_of_people = len(self.images_final)
        self.num_of_image = len(self.images_final[0])

    # Method 2: get heat maps
    def CenterGaussianHeatMap(self, c_x, c_y):
        start = time.time()
        img_height = self.height
        img_width = self.width
        variance = 21
        gaussian_map = np.zeros((img_height, img_width))
        for x_p in range(img_width):
            for y_p in range(img_height):
                dist_sq = (x_p - c_x) * (x_p - c_x) + \
                          (y_p - c_y) * (y_p - c_y)
                exponent = dist_sq / 2.0 / variance / variance
                gaussian_map[y_p, x_p] = np.exp(-exponent)
        end = time.time()
        print("single heat_map time:%d" % (end - start))
        # test heatmap
        cv2.imwrite('./a.png', gaussian_map[:, :] * 255)
        return gaussian_map

    def generate_heatmap(self, joints):
        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'
        if self.target_type == 'gaussian':
            heatmaps = [[[] for i in range(self.num_of_image)] for i in range(self.num_of_people)]
            for people_id in range(self.num_of_people):
                for img_id in range(self.num_of_image):
                    for joint_id in range(self.num_joints):
                        c_x = joints[people_id][img_id][joint_id][0]
                        c_y = joints[people_id][img_id][joint_id][1]
                        single_gaussian_map = self.CenterGaussianHeatMap(c_x, c_y)
                        heatmaps[people_id][img_id].append(single_gaussian_map)
        return heatmaps

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 2]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight


    def get_heatmaps(self, joints):
        heatmaps = [[[] for i in range(self.num_of_image)] for i in range(self.num_of_people)]
        heatmaps_weight = [[[] for i in range(self.num_of_image)] for i in range(self.num_of_people)]
        joints = np.array(joints)
        for people_id in range(self.num_of_people):
            for img_id in range(self.num_of_image):
                target, target_weight = self.generate_target(joints[people_id][img_id],joints[people_id][img_id])
                heatmaps[people_id][img_id].append(target)
                heatmaps_weight[people_id][img_id].append(target_weight)

        return heatmaps,heatmaps_weight


