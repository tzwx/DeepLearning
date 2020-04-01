# @Time    : 2020/1/13 15:55
# @Author  : FRY--
# @FileName: loss.py
# @Software: PyCharm
# @Blog    ：https://fryddup.github.io

import torch.nn as nn
import numpy as np

np.set_printoptions(threshold=np.inf)

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight
    def forward(self, output, target, target_weight):
        assert output.shape == target.shape ,"数据维度不相同！"
        batch_size = output.size(0)
        num_joints = output.size(1)
        # heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        # heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for batch_idx in range(batch_size):
            for idx in range(num_joints):
                heatmap_prediction = output[batch_idx][idx]
                heatmap_gt = target[batch_idx][idx]
                if self.use_target_weight:
                    target_weight_current = target_weight[batch_idx][idx]
                    loss += 0.5 * self.criterion(heatmap_prediction.mul(target_weight_current),
                                                 heatmap_gt.mul(target_weight_current))
                else:
                    loss += 0.5 * self.criterion(heatmap_prediction, heatmap_gt)

        return loss / num_joints
