# @Time    : 2019/12/12 15:20
# @Author  : FRY--
# @FileName: DataLoader.py
# @Software: PyCharm
# @Blog    ：https://fryddup.github.io

import cv2
import numpy as np
import os
import json
import lib.utils as utils
import torch
# Split Video Frames
class Videodata(object):
    def __init__(self, path):

        self.path = path
        cap = cv2.VideoCapture(path)
        assert cap.isOpened(), 'Cannot capture source'
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.datalen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        path_name, tempname = os.path.split(path)
        (filename, extension) = os.path.splitext(tempname)
        self.filename = filename
        self.path_name = path_name

    def update(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened(), "Can't find video"
        img = []
        for i in range(self.datalen):
            (flag, frame) = cap.read()
            filename = str(self.path_name) + str(self.filename) + str(i) + '.jpg'
            if flag == True:
                cv2.imwrite(filename, frame)
                # img.append(frame)

    def get_Info(self):
        return (self.fps, self.width, self.height)

    def length(self):
        return self.datalen

# get Img_info _ windows
class Imgdata_first(object):
    def __init__(self, path,floder_path):
        self.json_path = path  # 传入json文件路径
        self.meat_data = json.loads(open(path, encoding='utf-8').read())

    def get_bbox(self):
        data = self.meat_data
        bbox = [[] for i in range(len(data))]
        for i in range(len(data)):
            content = data[i]['candidates']
            for j in range(len(content)):
                bbox[i].append(content[j]['det_bbox'])
        return bbox

    # get img name Info,与bbox一一对应
    def get_img(self, bbox):
        images = []
        img_path, _ = os.path.split(self.json_path)
        dir_list = os.listdir(img_path)
        # if len(bbox) == (len(dir_list) + 1):
        for i in range(len(bbox)):
            if i < 10:
                imgpath = img_path + '/' + 'frame%05d' % (i) + '.jpg'
            elif i >= 10 and i <= 99:
                imgpath = img_path + '/' + 'frame%05d' % (i) + '.jpg'
            images.append(imgpath)

        return images
#get img_gt_box Info _ Linux
class Imgdata(object):

    def __init__(self,json_path,floder_path,index):
        self.index = index
        self.json_path = json_path
        self.floder_path = floder_path
        self.meta_data = json.loads(open(self.json_path, encoding='utf-8').read())
        self.data = self.meta_data[self.index]


    def get_Info(self):
        length = len(self.data)
        img = [[] for i in range(length)]
        bbox = [[] for i in range(length)]
        key_gt = [[] for i in range(length)]
        for i in range(length):
            img_path = self.data[i]['image']['folder']
            path = str(self.floder_path + img_path + '/' + self.data[i]['image']['name'])
            img[i].append(path)
            coord = []
            if len(self.data[i]['candidates']) == 0:
                coord = []
                bbox[i].append(coord)
                key_gt[i].append(coord)
            else:
                coord.append(self.data[i]['candidates'])
                for j in range(len(coord[0])):
                    bbox[i].append(coord[0][j]['det_bbox'])
                    key_gt[i].append(coord[0][j]['pose_keypoints_2d'])
        return img,bbox,key_gt,img_path

    def data_adjustment(self,keypoint):
        length = len(keypoint)
        keypoint_correct = [ []for i in range(length) ]
        temp = []
        for i in range(length): #photo
            if(len(keypoint[i])) == 1:
                temp = []
                keypoint_correct[i].append(temp)
            else:
                cell_length = len(keypoint[i])
                for j in range(cell_length): #people
                    temp1 = [ []for i in range(cell_length) ]
                    for k in range(15):
                        if k == 0:
                            temp1[j].append(keypoint[i][j][0:3])
                        else:
                            temp1[j].append(keypoint[i][j][k*3:k*3+3])
                    keypoint_correct[i].append(temp1[j])

        return keypoint_correct

class Data_pipline(object):

    def __init__(self, index):
        self.index = index

    def genCenterMap(self, x, y, sigma, size_w, size_h):
        """
        generate Gaussian heat map
        :param x: center point
        :param y: center point
        :param sigma:
        :param size_w: image width
        :param size_h: image height
        :return:            numpy           w * h
        """
        gridy, gridx = np.mgrid[0:size_h, 0:size_w]
        D2 = (gridx - x) ** 2 + (gridy - y) ** 2
        return np.exp(-D2 / 2.0 / sigma / sigma)  # numpy 2d

    def update(self):
        img = Imgdata('dataset/json_data/posetrack_train.json',
                      '/media/jion/D/chenhaoming/DataSet/PoseTrack2017/posetrack_data/',
                      self.index)
        images, bbox, key_gt, floder_path = img.get_Info()
        im = images
        gt_key = img.data_adjustment(key_gt)
        path_current = '/media/jion/D/chenhaoming/DataSet/PoseTrack2017/posetrack_data/' + floder_path + '/'
        trans_martix = utils.crop_img(path_current=path_current,images=images,bbox=bbox) #Generate new data  in origin floder
        #reshape data
        gt_key_mid = []
        trans = []
        bboxes = []
        # heatmaps_after = []
        imgs_after = [[] for i in range(len(images))]
        trans_img = []
        for m in range(len(images)):
            if len(bbox[m][0]) != 0:
                for num_box in range(len(bbox[m])):
                    imgs_after[m].append(path_current + 'transform_%d_%d.png' % (m, num_box))
        # GET singe_people
        for img_index in range(len(images)):
            # data = cv2.imread()
            if len(trans_martix[img_index]) > 0:
                # heatmaps_after.append(data_heatmaps[img_index])
                trans_img.append(imgs_after[img_index])
                gt_key_mid.append(gt_key[img_index])
                trans.append(trans_martix[img_index])
                bboxes.append(bbox[img_index])

        # todo The number og people need to adjust  get  minimal ok
        numbers_of_people = []
        for i in range(len(trans_img)):
            numbers_of_people.append(len(trans_img[i]))

        # print(len(trans_img))
        num_people = min(numbers_of_people)
        img_final = [[] for i in range(num_people)]
        # heatmap_final = [[] for i in range(num_people)]
        key_final = [[] for i in range(num_people)]
        trans_final = [[] for i in range(num_people)]
        bbox_final = [[] for i in range(num_people)]


        for people_index in range(num_people):
            for img_idx in range(len(trans_img)):
                img_final[people_index].append(trans_img[img_idx][people_index])
                # heatmap_final[people_index].append(heatmaps_after[img_idx][people_index])
                key_final[people_index].append(gt_key_mid[img_idx][people_index])
                trans_final[people_index].append(trans[img_idx][people_index])
                bbox_final[people_index].append(bboxes[img_idx][people_index])
        # todo delete destory image
        img_des = []
        for people_index in range(num_people):
            for img_idx in range(len(img_final[0])):
                data =  cv2.imread(img_final[people_index][img_idx])
                if data is None:
                    img_des.append(img_idx)
        img_des_final = list(set(img_des))
        img_des_final.sort()

        count = 0
        if len(img_des) > 0:
            for i in range(len(img_des_final)):
                for people_index in range(num_people):
                    idx = img_des_final[i] - count
                    del img_final[people_index][idx]
                    del key_final[people_index][idx]
                    del bbox_final[people_index][idx]
                    # del trans_final[people_index][idx]
                count = count + 1

        # for people_index in range(num_people):
        #     for img_idx in range(len(img_final[0])):

        # del img_final[people_index][img_idx]
        # del key_final[people_index][img_idx]

        #joints
        Get_joints = utils.Joints_data(images,img_final)

        #coordinates transform
        key_new = [[[] for i in range(len(img_final[0]))]for i in range(num_people)]
        for people_index in range(num_people):
            for img_idx in range(len(bbox_final[0])):
                for joint in range(15):
                    coordinates = np.array(key_final[people_index][img_idx][joint])
                    correct = coordinates[-1]
                    new_coordinates = np.dot(trans_final[people_index][img_idx], coordinates)
                    new_coordinates = new_coordinates.tolist()
                    new_coordinates.append(correct)

                    key_new[people_index][img_idx].append(np.array(new_coordinates))
                    # key_new[people_index][img_idx][joint].append(correct)

        heatmaps,heatmaps_weight = Get_joints.get_heatmaps(joints=key_new)

        center_map = self.genCenterMap(x=184, y=184, sigma=21, size_w=368, size_h=368)
        center_map = torch.Tensor(center_map)
        center_map = center_map.unsqueeze(0)
        center_map = center_map.unsqueeze(0)

        return img_final, heatmaps, heatmaps_weight, center_map, bbox_final, im, key_new












