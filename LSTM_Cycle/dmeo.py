# @Time    : 2020/3/17 上午 10:50
# @Author  : FRY--
# @FileName: dmeo.py
# @Software: PyCharm
# @Blog    ：https://fryddup.github.io

import test.demo_model
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import os
from dataset.DataLoader import Data_pipline
from tqdm import tqdm
from lib.utils import get_final_preds
from lib.utils import _box2cs

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model = test.demo_model.LSTM_Cycle_Test()
pre_train = torch.load('./models_new_mid_loss/lstm_epoch_7Iteration_180_.pth')
model.load_state_dict(pre_train)
model.cuda()
data = Data_pipline(0)
img, heatmaps, heatmaps_weight, centermap, bbox, im, _= data.update()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
img_data_final = []
cycle_length = len(img[0])
heatmaps_model_forward = [[] for i in range(len(img))]
heatmaps_model_backward = [[] for i in range(len(img))]
heatmaps_predict = [[] for i in range(len(img))]


over_people = cycle_length % 5

if over_people == 1:
    flag = 0
else:
    flag = 1

if __name__ == '__main__':
    for people_index in tqdm(range(len(img))):
        numbers = len(img[0]) // 5 + 1
        for num_cycle in range(numbers):
            if num_cycle != numbers - 1:
                images = img[people_index][num_cycle * 5: num_cycle * 5 + 5]  # 0->10 10->20 20->30
                for i in range(5):
                    data_np = cv2.imread(images[i], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                    input = transform(data_np)
                    # img = tran(data_np)
                    img_data_final.append(input)
                heatmaps_Groungtruth = heatmaps[people_index][num_cycle * 5: num_cycle * 5 + 5]
                heatmaps_weight_Groungtruth = heatmaps[people_index][num_cycle * 5: num_cycle * 5 + 5]

            # over_people > 1 Calculate
            elif num_cycle == numbers - 1 and over_people > 1:
                images = img[people_index][cycle_length - 5: cycle_length]
                for i in range(5):
                    data_np = cv2.imread(images[i], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                    input = transform(data_np)
                    img_data_final.append(input)
                heatmaps_Groungtruth = heatmaps[people_index][cycle_length - 5: cycle_length]
                heatmaps_weight_Groungtruth = heatmaps_weight[people_index][cycle_length - 5:cycle_length]

            if flag == 1 or num_cycle != numbers - 1:
                # calculate
                heatmaps_pass = []
                heatmaps_pass.append(heatmaps_Groungtruth[0][0])
                heatmaps_pass.append(heatmaps_Groungtruth[-1][-1])

                model.eval()
                with torch.no_grad():
                    heatmaps_forward, heatmaps_backward = model(img_data_final, heatmaps_pass, centermap)
                if num_cycle != numbers - 1:
                    for i in range(5):
                        heatmaps_model_forward[people_index].append(heatmaps_forward[i])
                        heatmaps_model_backward[people_index].append(heatmaps_backward[i])
                else:
                    heatmaps_forward_end = heatmaps_forward[-over_people:-1]
                    heatmaps_backward_end = heatmaps_backward[-over_people:-1]
                    for i in range(over_people - 1):
                        heatmaps_model_forward[people_index].append(heatmaps_forward[i])
                        heatmaps_model_backward[people_index].append(heatmaps_backward[i])

                # for i in range(len(heatmaps_model_forward)):
                #     new_heatmaps = (heatmaps_model_forward[people_index][i] +
                #                     heatmaps_model_backward[people_index][i]) / 2
                #     heatmaps_predict[people_index].append(new_heatmaps)

print("done!")
im = im[55:85]
# vis
for people_index in range(len(img)):
    del img[people_index][-1]
    del bbox[people_index][-1]
for people_index in tqdm(range(len(img))):
    for img_idx in range(len(img[0])):
        ima= im[img_idx]
        image = cv2.imread(ima[0])
        heatmap = heatmaps_model_backward[people_index][img_idx]
        # cv2.imwrite('./heatmap.jpg',heatmap.cpu().detach().numpy()[5][0, :, :] * 255)
        c, s = _box2cs(bbox[people_index][img_idx], 720, 1280)
        # compute coordinate
        preds, maxvals = get_final_preds(
            heatmap.clone().cpu().numpy(), np.asarray([c]), np.asarray([s]))

        image = image.copy()
        for mat in preds[0]:
            x, y = int(mat[0]), int(mat[1])
            cv2.circle(image, (x, y), 2, (255, 0, 0), 2)

        cv2.imwrite('./testb/img_' + str(people_index) + "_" + str(img_idx) + '_' + '.jpg',image)
        # for i in range(15):
        #     # heatmap
        #     cv2.imwrite('./test_heatmap/heatmap_' + str(0) + "_" + str(0) + '_' + str(i) + '.jpg',
        #                 heatmap.cpu().detach().numpy()[0][i, :, :] * 255)


# test
# for i in range(15):
#     heatmap_new = cv2.imread('./test_heatmap/heatmap_'+str(0)+"_"+ str(0)+ '_'+str(i) + '.jpg')
#     heatmap_new = cv2.resize(heatmap_new,(368,368))
#     heatmap = cv2.applyColorMap(heatmap_new, cv2.COLORMAP_JET)
#     # cv2.imwrite('./heat.jpg', heatmap)
#     superimposed_img = heatmap_new + image*0.5
#     cv2.imwrite('./compose.jpg', superimposed_img)
#     newimg = cv2.imread('./compose.jpg')
#     image = newimg
# superimposed_img = cv2.imread('./compose.jpg')
# image = np.array(superimposed_img,np.uint8)
# superimposed_img = cv2.applyColorMap(image, cv2.COLORMAP_JET)
# cv2.imwrite('./compose.jpg', superimposed_img)
# print('s')


# heatmap process

# todo
#  find a new bug: Cycle length
