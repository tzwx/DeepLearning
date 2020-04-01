# @Time    : 2020/1/12 14:26
# @Author  : FRY--
# @FileName: train.py.py
# @Software: PyCharm
# @Blog    ：https://fryddup.github.io

from dataset import DataLoader
import cv2
import numpy as np
import torchvision.transforms as transforms
import torch
import net.loss as Loss
from net.Lstm import LSTM_Cycle
import torch.optim as optim
import os
import warnings
warnings.filterwarnings("ignore")
EPOCHS = 10
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# np.set_printoptions(threshold=np.inf)
class trainer():
    def __init__(self,model,index):
        self.index = index
        self.data = DataLoader.Data_pipline(self.index)
        img, heatmaps, heatmaps_weight, centermap, bbox, im, _ = self.data.update()
        self.images = img
        self.heatmaps = heatmaps
        self.heatmaps_weight = heatmaps_weight
        self.center_map = centermap
        self.cycle_length = len(self.images[0])  # 图片数量
        self.num_of_people = len(self.images)
        self.num_of_image = len(self.images[0])
        self.model = model

    def get_traindata(self, images, heatmaps):
        pass

    def train(self):
        img_data_final = []
        for people_index in range(self.num_of_people):
            print("总人数：%d, 当前是第 %d 个人" % (self.num_of_people, people_index+1))
            # if self.cycle_length % 5 != 0:  #
            numbers = self.cycle_length // 5 + 1  # 38张图循环8次
            for num_cycle in range(numbers):  # ->8
                if num_cycle != numbers - 1:  # 前7组 0-5-10-15-20-25-30-35-38
                    images = self.images[people_index][num_cycle * 5: num_cycle * 5 + 5]  # 0->10 10->20 20->30
                    for i in range(5):
                        data_np = cv2.imread(images[i], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                        tran = transforms.ToTensor()
                        img = tran(data_np)
                        img_data_final.append(img)
                    heatmaps_Groungtruth = self.heatmaps[people_index][num_cycle * 5: num_cycle * 5 + 5]
                    heatmaps_weight_Groungtruth = self.heatmaps[people_index][num_cycle * 5: num_cycle * 5 + 5]
                else:
                    images = self.images[people_index][self.cycle_length - 5: self.cycle_length]
                    for i in range(5):
                        data_np = cv2.imread(images[i], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                        tran = transforms.ToTensor()
                        img = tran(data_np)
                        img_data_final.append(img)
                    # img_data_final.cuda()
                    heatmaps_Groungtruth = self.heatmaps[people_index][self.cycle_length - 5: self.cycle_length]
                    heatmaps_weight_Groungtruth = self.heatmaps_weight[people_index][
                                                  self.cycle_length - 5: self.cycle_length]
                # img_data_final = img_data_final.cuda()
                # 1,15,45,45
                heatmaps_pass = []
                heatmaps_pass.append(heatmaps_Groungtruth[0][0])
                heatmaps_pass.append(heatmaps_Groungtruth[-1][0])
                heatmap_0 = []
                heatmap_1 = []
                heatmap_2 = []
                for index in range(1,4):
                    heatmaps_start, heatmaps_end, heatmap_final = self.model(img_data_final, index, heatmaps_pass, self.center_map)
                    heatmap_0.append(heatmaps_start[0])
                    heatmap_1.append(heatmaps_end[0])
                    heatmap_2.append(heatmap_final)
                # for i in range(5):
                #     heatmap_current = (heatmap_2[0][i] + heatmap_2[1][i] + heatmap_2[2][i])/3
                #     heatmap_3.append(heatmap_current)
                optimizer = optim.Adam(model.parameters(), lr=0.0001)
                optimizer.zero_grad()
                loss = Loss.JointsMSELoss(False)
                heatmaps_start_tensor = torch.cat([heatmap_0[0], heatmap_0[1],heatmap_0[2],], dim=0)
                heatmaps_end_tensor = torch.cat([heatmap_1[0], heatmap_1[1],heatmap_1[2],], dim=0)
                output = torch.cat([heatmaps_start_tensor, heatmaps_end_tensor], dim=0)
                heatmaps_gt_start_tensor = torch.cat([torch.Tensor(heatmaps_Groungtruth[0]),
                                                     torch.Tensor(heatmaps_Groungtruth[0]),
                                                     torch.Tensor(heatmaps_Groungtruth[0])], dim=0)
                heatmaps_gt_end_tensor = torch.cat([torch.Tensor(heatmaps_Groungtruth[-1]),
                                                    torch.Tensor(heatmaps_Groungtruth[-1]),
                                                    torch.Tensor(heatmaps_Groungtruth[-1])], dim=0)
                target = torch.cat([heatmaps_gt_start_tensor, heatmaps_gt_end_tensor], dim=0).cuda()
                heatmaps_weight_gt_start_tensor = torch.cat([torch.Tensor(heatmaps_weight_Groungtruth[0]),
                                                      torch.Tensor(heatmaps_weight_Groungtruth[0]),
                                                      torch.Tensor(heatmaps_weight_Groungtruth[0])], dim=0)
                heatmaps_weight_gt_end_tensor = torch.cat([torch.Tensor(heatmaps_weight_Groungtruth[-1]),
                                                    torch.Tensor(heatmaps_weight_Groungtruth[-1]),
                                                    torch.Tensor(heatmaps_weight_Groungtruth[-1])], dim=0)
                target_weight = torch.cat(
                    [heatmaps_weight_gt_start_tensor, heatmaps_weight_gt_end_tensor], dim=0).cuda()
                loss_value = 0
                loss_value_start_end = loss(output, target, target_weight)
                loss_value += loss_value_start_end
                heatmap_tensor_all  = torch.cat([torch.Tensor(heatmaps_Groungtruth[0]),
                                                 torch.Tensor(heatmaps_Groungtruth[1]),
                                                 torch.Tensor(heatmaps_Groungtruth[2]),
                                                 torch.Tensor(heatmaps_Groungtruth[3]),
                                                 torch.Tensor(heatmaps_Groungtruth[4])],dim=0).cuda()
                heatmap_weight_tensor_all = torch.cat([torch.Tensor(heatmaps_weight_Groungtruth[0]),
                                                       torch.Tensor(heatmaps_weight_Groungtruth[1]),
                                                       torch.Tensor(heatmaps_weight_Groungtruth[2]),
                                                       torch.Tensor(heatmaps_weight_Groungtruth[3]),
                                                       torch.Tensor(heatmaps_weight_Groungtruth[4])], dim=0).cuda()
                heatmap_tensor_1 = torch.cat([ heatmap_2[0][0], heatmap_2[0][1], heatmap_2[0][2],
                                               heatmap_2[0][3], heatmap_2[0][4] ],dim=0).cuda()
                heatmap_tensor_2 = torch.cat([ heatmap_2[1][0], heatmap_2[1][1], heatmap_2[1][2],
                                               heatmap_2[1][3], heatmap_2[1][4] ],dim=0).cuda()
                heatmap_tensor_3 = torch.cat([ heatmap_2[2][0], heatmap_2[2][1], heatmap_2[2][2],
                                               heatmap_2[2][3], heatmap_2[2][4] ],dim=0).cuda()
                loss_1 = loss(heatmap_tensor_1, heatmap_tensor_all, heatmap_weight_tensor_all)
                loss_2 = loss(heatmap_tensor_2, heatmap_tensor_all, heatmap_weight_tensor_all)
                loss_3 = loss(heatmap_tensor_3, heatmap_tensor_all, heatmap_weight_tensor_all)
                loss_value += ( (loss_1 + loss_2 + loss_3) / 3 )
                print("Current Cycle：%d, LOSS: %f" % (num_cycle, loss_value.cpu().detach().numpy()) )
                # print("当前cycle：%d , loss: %d" %(num_cycle, loss_value.item()) )
                loss_value.backward()
                optimizer.step()


model = LSTM_Cycle().cuda()
for epoch in range(EPOCHS):
    for i in range(250):
        print("Epoch: %d, Iteration: %d" % (epoch, i))
        train_c = trainer(model, i)
        train_c.train()
        del train_c
        #save model
        if i >= 10 and i % 10 == 0:
            path = './models_new_mid_loss/lstm_' + 'epoch_' + str(epoch) +'Iteration_' + str(i)+'_.pth'
            torch.save(model.state_dict(), path)

#
# train_c = trainer(model, 43)
# train_c.train()

#
# Data 205
# net 250
# train 114
# loss 27
# util 254