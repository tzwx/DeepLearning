# @Time    : 2020/2/29 17:31
# @Author  : FRY--
# @FileName: train.py
# @Software: PyCharm
# @Blog    ：https://fryddup.github.io

from data_loader import  get_img
from net.net import Generator
from net.net import Discriminator
import torch
import torch.nn as nn
import cv2
import numpy as np
import torchvision.transforms as transforms

BATCH_SIZE = 100
G_LEARNING_RATE = 0.0001
D_LEARNING_RATE = 0.0001
EPOCHS = 2500

trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ]
)
# Instance
g = Generator().cuda()
d = Discriminator().cuda()

# define loss
g_loss = nn.BCELoss() # Binary crossEntropy
d_loss = nn.BCELoss()

# define optimizer
g_optimizer = torch.optim.Adam(g.parameters(),lr=G_LEARNING_RATE)
d_optimizer = torch.optim.Adam(d.parameters(),lr=D_LEARNING_RATE)

# define labels
label_real = torch.ones(BATCH_SIZE).cuda()
label_fake = torch.zeros(BATCH_SIZE).cuda()

#all images
images_truth = get_img()
data_length = len(images_truth)

def train():
    for epoch in range(EPOCHS):
        # shuffle
        np.random.shuffle(images_truth)
        images_real_loader = []
        count = 0
        for index in range(data_length):
            count = count + 1
            images_real_loader.append(trans(images_truth[index]).numpy())
            # images_real_loader[100,3,96,96]
            if count == BATCH_SIZE:
                count = 0 # reset

                # Train Discriminator
                # train real data to gpu
                # if real -> d(real_img) = 1
                images_real_loader_tensor = torch.Tensor(images_real_loader)
                images_real_loader_tensor = images_real_loader_tensor.permute(0,3,1,2).cuda()
                images_real_loader.clear()
                # graddient _ zero
                d_optimizer.zero_grad()
                # real image output_66
                realimage_d = d(images_real_loader_tensor).squeeze() # descent  [100，1，1，1] -> [100,1]
                # loss
                d_realimg_loss = d_loss(realimage_d, label_real)
                # loss backward
                d_realimg_loss.backward()

                # train generate data
                # if generator d(generate_img) = 0
                images_fake_loader = torch.randn(BATCH_SIZE, 100, 1, 1).cuda()
                # detach() g no gradient -> fix generator
                images_fake_loader_tensor = g(images_fake_loader).detach()
                fakeimg_d = d(images_fake_loader_tensor).squeeze()
                d_fakeimg_loss = d_loss(fakeimg_d, label_fake)
                d_fakeimg_loss.backward()

                d_optimizer.step()

                # Train Generator
                fake_data = torch.randn(BATCH_SIZE, 100, 1, 1).cuda()
                g_optimizer.zero_grad()
                generator_images = g(fake_data)
                generator_images_score = d(generator_images).squeeze()
                gen_loss = g_loss(generator_images_score, label_real)
                gen_loss.backward()
                g_optimizer.step()

                print("Current epoch:%d, Iteration: %d, Discriminator Loss: %f, Generator Loss: %f"
                      % (epoch, (index//BATCH_SIZE)+1,
                         (d_realimg_loss+d_fakeimg_loss).detach().cpu().numpy(),
                         gen_loss.detach().cpu().numpy()))

                torch.save(g, "pkl"+str(epoch)+"generator.pkl")

if __name__ == "__main__":
    train()