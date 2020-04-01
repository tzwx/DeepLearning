import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class LSTM_Cycle(nn.Module):
    def __init__(self, outclass=15, T=5):  # outclass 输出的通道数；T为区间间隔
        super(LSTM_Cycle, self).__init__()
        self.outclass = outclass
        self.T = T

        self.conv_up_down = nn.Conv2d(15, 32, 1, 1)
        self.pool_center_lower = nn.AvgPool2d(kernel_size=9, stride=8)

        # conv_net_heatmap
        self.conv1_convnet1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)  # 3 * 368 * 368
        self.pool1_convnet1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2_convnet1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool2_convnet1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3_convnet1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool3_convnet1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv4_convnet1 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv5_convnet1 = nn.Conv2d(32, 512, kernel_size=9, padding=4)
        self.conv6_convnet1 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv7_convnet1 = nn.Conv2d(512, self.outclass, kernel_size=1)  # 512 * 45 * 45

        # conv_net_img_feature
        self.conv1_convnet2 = nn.Conv2d(3, 128, kernel_size=9, padding=4)  # 3 * 368 * 368
        self.pool1_convnet2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2_convnet2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)  # 128 * 184 * 184
        self.pool2_convnet2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3_convnet2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)  # 128 * 92 * 92
        self.pool3_convnet2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv4_convnet2 = nn.Conv2d(128, 32, kernel_size=5, padding=2)  # 32 * 45 * 45

        # convNet_relate_heatmap
        self.Mconv1_convnet3 = nn.Conv2d(48, 128, kernel_size=11, padding=5)
        self.Mconv2_convnet3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_convnet3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_convnet3 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_convnet3 = nn.Conv2d(128, self.outclass, kernel_size=1, padding=0)

        # '''
        # lstm_cycle
        # 32是每帧heatmap通道数（网络决定，暂定15）
        # 1为background
        # outclass为上一time输出的
        # '''
        self.conv_ix_lstm = nn.Conv2d(32 + self.outclass + 1, 48, kernel_size=3, padding=1, bias=True)
        self.conv_ih_lstm = nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False)

        self.conv_fx_lstm = nn.Conv2d(32 + self.outclass + 1, 48, kernel_size=3, padding=1, bias=True)
        self.conv_fh_lstm = nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False)

        self.conv_ox_lstm = nn.Conv2d(32 + self.outclass + 1, 48, kernel_size=3, padding=1, bias=True)
        self.conv_oh_lstm = nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False)

        self.conv_gx_lstm = nn.Conv2d(32 + self.outclass + 1, 48, kernel_size=3, padding=1,
                                      bias=True)  # 15 + 32 = 47
        self.conv_gh_lstm = nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False)
        # initial lstm_cycle
        self.conv_gx_lstm0 = nn.Conv2d(32 + self.outclass +1, 48, kernel_size=3, padding=1)
        self.conv_ix_lstm0 = nn.Conv2d(32 + self.outclass +1, 48, kernel_size=3, padding=1)
        self.conv_ox_lstm0 = nn.Conv2d(32 + self.outclass +1, 48, kernel_size=3, padding=1)

    def covNet_initial_heatmap(self, img):
        '''
        :这块用不到
        :return:
        '''
        img = img.unsqueeze(0).permute(0,3,1,2)
        x = self.pool1_convnet1(F.relu(self.conv1_convnet1(img)))  # output 128 * 184 * 184
        x = self.pool2_convnet1(F.relu(self.conv2_convnet1(x)))  # output 128 * 92 * 92
        x = self.pool3_convnet1(F.relu(self.conv3_convnet1(x)))  # output 128 * 45 * 45
        x = F.relu(self.conv4_convnet1(x))  # output 32 * 45 * 45
        x = F.relu(self.conv5_convnet1(x))  # output 512 * 45 * 45
        x = F.relu(self.conv6_convnet1(x))  # output 512 * 45 * 45
        initial_heatmap = self.conv7_convnet1(x)  # output class * 45 * 45
        return initial_heatmap

    def convNet_featuremap(self, image):
        '''
        :param image: 3 * 368 * 368
        :return: Fs(.) features 32 * 45 * 45
        '''
        # image=image.unsqueeze(0)
        # ok le
        image = image.unsqueeze(0).cuda()
        x = self.pool1_convnet2(F.relu(self.conv1_convnet2(image)))  # output 128 * 184 * 184  192*192
        x = F.relu(self.pool2_convnet2(F.relu(self.conv2_convnet2(x))))  # output 128 * 92 * 92 96*96
        x = self.pool3_convnet2(F.relu(self.conv3_convnet2(x)))  # output 128 * 45 * 45
        x = F.relu(self.conv4_convnet2(x))  # output 32 * 45 * 45
        return x  # output 32 * 45 * 45 [1,32,45,45]

    def convNet_heatmap(self, hide_t):
        '''
        :此层结构是又lstm计算的结果输出最终的 heatmaps
        :param hide_t: 48 * 45 * 45
        :return: heatmap outclass * 45 * 45
        '''
        x = F.relu(self.Mconv1_convnet3(hide_t))  # output 128 * 45 * 45
        x = F.relu(self.Mconv2_convnet3(x))  # output 128 * 45 * 45
        x = F.relu(self.Mconv3_convnet3(x))  # output 128 * 45 * 45
        x = F.relu(self.Mconv4_convnet3(x))  # output 128 * 45 * 45
        x = self.Mconv5_convnet3(x)  # output (class) * 45 * 45
        return x  # heatmap (class+1) * 45 * 45

    def lstm(self, heatmap_prior,  features, center_map, hide_t_1, cell_t_1):
        '''

        :param heatmap_prior:
        :param heatmap_t:
        :param hide_t_1: 48 * 45 * 45
        :param cell_t_1: 48 * 45 * 45
        :return:
        hide_t: 48 * 45 * 45
        cell_t: 48 * 45 * 45
        '''
        heatmap_prior = heatmap_prior.cuda()
        xt = torch.cat([heatmap_prior, features, center_map], dim=1)  # (32 + (class+1)*2) * 45 * 45

        gx = self.conv_gx_lstm(xt)  # output: 48 * 45 * 45
        gh = self.conv_gh_lstm(hide_t_1)  # output: 48 * 45 * 45
        g_sum = gx + gh
        gt = F.tanh(g_sum)

        ox = self.conv_ox_lstm(xt)  # output: 48 * 45 * 45
        oh = self.conv_oh_lstm(hide_t_1)  # output: 48 * 45 * 45
        o_sum = ox + oh
        ot = F.sigmoid(o_sum)

        ix = self.conv_ix_lstm(xt)  # output: 48 * 45 * 45
        ih = self.conv_ih_lstm(hide_t_1)  # output: 48 * 45 * 45
        i_sum = ix + ih
        it = F.sigmoid(i_sum)

        fx = self.conv_fx_lstm(xt)  # output: 48 * 45 * 45
        fh = self.conv_fh_lstm(hide_t_1)  # output: 48 * 45 * 45
        f_sum = fx + fh
        ft = F.sigmoid(f_sum)

        cell_t = ft * cell_t_1 + it * gt
        hide_t = ot * F.tanh(cell_t)

        return cell_t, hide_t

    def lstm0(self, x):
        gx = self.conv_gx_lstm0(x)
        ix = self.conv_ix_lstm0(x)
        ox = self.conv_ox_lstm0(x)

        gx = F.tanh(gx)
        ix = F.sigmoid(ix)
        ox = F.sigmoid(ox)

        cell1 = F.tanh(gx * ix)
        hide_1 = ox * cell1
        return cell1, hide_1

    def frameStart(self, img, heatmap, center_map):
        '''
        Start Frame
        :param img: 3 * 368 * 368
        :param heatmap:  out_class * 45 * 45
        :return:
        heatmap:                     original
        cell_t:                      48 * 45 * 45
        hide_t:                      48 * 45 * 45
        '''
        heatmap = heatmap.cuda()
        feature = self.convNet_featuremap(img) # 计算特征图
        center_map = self.pool_center_lower(center_map).cuda() # 计算置信图
        x = torch.cat([heatmap, feature, center_map], dim=1)
        cell1, hide1 = self.lstm0(x)
        return cell1, hide1

    def frameNormal(self, img,  heatmap_prior, cmap,  cell_t_1, hide_t_1):
        '''

        :param img:  3 * 368 * 368
        :param heatmap:
        :param cell_t_1:
        :param hide_t_1:
        :return:
        '''
        features_t = self.convNet_featuremap(img)
        centermap_t = self.pool_center_lower(cmap).cuda()
        cell_t, hide_t = self.lstm(heatmap_prior,  features_t, centermap_t, cell_t_1, hide_t_1)
        new_heatmap = self.convNet_heatmap(hide_t)
        return new_heatmap, cell_t, hide_t

    def forward(self, images, index, heatmaps_start_end_gt, center_map):
        '''

        :param img: frame img
        :param heatmap_: frame heatmap
        :return:
        '''
        img = images[0]
        heatmaps_init = heatmaps_start_end_gt[0]
        heatmaps_init = torch.Tensor(heatmaps_init)
        heatmaps_init = heatmaps_init.unsqueeze(0)
        img_cycle_2 = images[-1]
        heatmap_cycle_2_init = heatmaps_start_end_gt[1]
        heatmap_cycle_2_init = torch.Tensor(heatmap_cycle_2_init)
        heatmap_cycle_2_init = heatmap_cycle_2_init.unsqueeze(0)
        heat_maps_start_loss = []
        heat_maps_end_loss = []
        cell, hide = self.frameStart(img, heatmaps_init, center_map)  # 第一帧信息
        cell_2, hide_2 = self.frameStart(img_cycle_2, heatmap_cycle_2_init, center_map)  # 尾帧信息
        heatmap_pre = heatmaps_init
        heatmap_pre_right = heatmap_cycle_2_init

        left_forward_mid_heatmap  = []
        left_backward_mid_heatmap = []
        right_forward_mid_heatmap  = []
        right_backward_mid_heatmap = []

        # forward_1
        i = index
        # Left 正反向传播
        # 正向 1
        for j in range(0, i + 1):
            img = images[j]  # 0,1
            heatmaps, cell, hide = self.frameNormal(img, heatmap_pre, center_map, cell, hide)
            left_forward_mid_heatmap.append(heatmaps) # 0,1
            heatmap_pre = heatmaps
        # backward_1
        heatmap_pre1 = heatmap_pre
        forward_list = [n for n in range(0, i)]
        forward_list.reverse()
        for k in forward_list:
            img = images[k]  # 0 (第一帧的heatmap作为输入直接计算第0帧的heatmap)
            heatmaps, cell, hide = self.frameNormal(img,  heatmap_pre1, center_map, cell, hide)
            left_backward_mid_heatmap.append(heatmaps)# 0
            heatmap_pre1 = heatmaps
        heat_maps_start_loss.append(heatmap_pre1)  # -> start_loss

        # Right 正向反向传播
        # backword 2 尾帧与起始frame相同，都是新的一轮Cycle ，故需要Initial
        back_list = [p for p in range(i, self.T)]
        back_list.reverse() # 4,3,2,1
        for index in back_list:
            img = images[index]
            heatmaps, cell, hide = self.frameNormal(img,  heatmap_pre_right, center_map, cell_2, hide_2)
            right_backward_mid_heatmap.append(heatmaps) # 4,3,2,1
            heatmap_pre_right = heatmaps # end 1

        # forward 2
        heatmap_pre_right_1 = heatmap_pre_right # 1
        for m in range(i+1, self.T):
            img = images[m]  # 2,3,4
            heatmaps, cell, hide = self.frameNormal(img, heatmap_pre_right_1, center_map,  cell, hide)
            right_forward_mid_heatmap.append(heatmaps)
            heatmap_pre_right_1 = heatmaps
        heat_maps_end_loss.append(heatmap_pre1)

        for i in range(len(right_forward_mid_heatmap)):
            left_forward_mid_heatmap.append(right_forward_mid_heatmap[i])

        forward_heatmap_loss = left_forward_mid_heatmap

        right_backward_mid_heatmap.reverse()
        left_backward_mid_heatmap.reverse()
        for i in range(len(right_backward_mid_heatmap)):
            left_backward_mid_heatmap.append(right_backward_mid_heatmap[i])

        backward_heatmap_loss = left_backward_mid_heatmap

        heatmap_final = []
        for i in range(5):
            heatmap_final.append( (forward_heatmap_loss[i] + backward_heatmap_loss[i]) / 2 )

        return heat_maps_start_loss, heat_maps_end_loss, heatmap_final


