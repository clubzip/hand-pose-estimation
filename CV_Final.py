# Various imports & settings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import torchvision
import tqdm
import cv2
from glob import glob
import pandas
import time

num_joints = 21
device = 'cuda:0'

# - For colab
# from google.colab import drive
# from google.colab.patches import cv2_imshow
# drive.mount('/content/gdrive')
# 2d pose estimator - pretrained
class CPM2DPose(nn.Module):
    def __init__(self):
        super(CPM2DPose, self).__init__()

        self.scoremap_list = []
        self.layers_per_block = [2, 2, 4, 2]
        self.out_chan_list = [64, 128, 256, 512]
        self.pool_list = [True, True, True, False]

        # MODEL 4 & 5
        self.relu = F.leaky_relu
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_1
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.batnorm1 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.batnorm2 = nn.BatchNorm2d(128)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.batnorm3 = nn.BatchNorm2d(256)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv4_3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv4_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv4_5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv4_6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv4_7 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.batnorm4 = nn.BatchNorm2d(256)
        self.conv5_1 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=True)  # conv0_2
        self.conv5_2 = nn.Conv2d(512, 21, kernel_size=1, stride=1, padding=0, bias=True)  # conv0_2
        self.conv6_1 = nn.Conv2d(149, 128, kernel_size=7, stride=1, padding=3, bias=True)  # conv0_2
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3, bias=True)  # conv0_2
        self.conv6_3 = nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3, bias=True)  # conv0_2
        self.conv6_4 = nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3, bias=True)  # conv0_2
        self.conv6_5 = nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3, bias=True)  # conv0_2
        self.conv6_6 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=True)  # conv0_2
        self.conv6_7 = nn.Conv2d(128, 21, kernel_size=1, stride=1, padding=0, bias=True)  # conv0_2
        self.batnorm6 = nn.BatchNorm2d(128)
        self.conv7_1 = nn.Conv2d(149, 128, kernel_size=7, stride=1, padding=3, bias=True)  # conv0_2
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3, bias=True)  # conv0_2
        self.conv7_3 = nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3, bias=True)  # conv0_2
        self.conv7_4 = nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3, bias=True)  # conv0_2
        self.conv7_5 = nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3, bias=True)  # conv0_2
        self.conv7_6 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=True)  # conv0_2
        self.conv7_7 = nn.Conv2d(128, 21, kernel_size=1, stride=1, padding=0, bias=True)  # conv0_2
        self.batnorm7 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout2d(p=0.1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.batnorm1(x)
        x = self.relu(self.conv1_2(x))
        x = self.maxpool(x)
        x = self.batnorm1(x)
        x = self.relu(self.conv2_1(x))
        x = self.batnorm2(x)
        x = self.relu(self.conv2_2(x))
        x = self.maxpool(x)
        x = self.batnorm2(x)
        x = self.relu(self.conv3_1(x))
        x = self.batnorm3(x)
        x = self.relu(self.conv3_2(x))
        x = self.batnorm3(x)
        x = self.relu(self.conv3_3(x))
        x = self.batnorm3(x)
        x = self.relu(self.conv3_4(x))
        x = self.maxpool(x)
        x = self.batnorm3(x)
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.batnorm4(x)
        x = self.relu(self.conv4_4(x))
        x = self.batnorm4(x)
        x = self.relu(self.conv4_5(x))
        x = self.batnorm4(x)
        x = self.relu(self.conv4_6(x))
        encoding = self.relu(self.conv4_7(x))
        x = self.relu(self.conv5_1(encoding))
        scoremap = self.conv5_2(x)

        x = torch.cat([scoremap, encoding], 1)
        x = self.relu(self.conv6_1(x))
        x = self.batnorm6(x)
        x = self.relu(self.conv6_2(x))
        x = self.batnorm6(x)
        x = self.relu(self.conv6_3(x))
        x = self.batnorm6(x)
        x = self.relu(self.conv6_4(x))
        x = self.batnorm6(x)
        x = self.relu(self.conv6_5(x))
        x = self.batnorm6(x)
        x = self.relu(self.conv6_6(x))
        scoremap = self.conv6_7(x)
        x = torch.cat([scoremap, encoding], 1)
        x = self.relu(self.conv7_1(x))
        x = self.batnorm7(x)
        x = self.relu(self.conv7_2(x))
        x = self.batnorm7(x)
        x = self.relu(self.conv7_3(x))
        x = self.batnorm7(x)
        x = self.relu(self.conv7_4(x))
        x = self.dropout(x)
        x = self.relu(self.conv7_5(x))
        x = self.batnorm7(x)
        x = self.relu(self.conv7_6(x))
        x = self.conv7_7(x)
        return x

class ObmanDataset(Dataset):
    def __init__(self, method=None):
        self.root = '/workspace/share/cv_final/dataset/' #Change this path
        self.x_data = []
        self.y_data = []

        if method == 'train':
            self.root = self.root + 'train/'
            self.img_path = sorted(glob(self.root + 'rgb/*.jpg'))

            for i in tqdm.tqdm(range(len(self.img_path))):
              # Read data and append to y_data
              num = self.img_path[i].split('.')[0].split('/')[-1]
              img_pkl = self.root + 'meta/' + str(num) + '.pkl'
              pkl = pandas.read_pickle(img_pkl)
              coords_2d = pkl['coords_2d']

              # Check if there are negative coordinates in the ground truth skeleton data and skip the image
              values = np.array(coords_2d)
              if(values.min() < 0):
                continue

              # ground truth
              self.y_data.append(coords_2d)
                
              # flipped ground truth
              coords_2d = np.array(coords_2d)
              a = 256 - coords_2d[:,1]
              coords_2d[:,1] = a
              self.y_data.append(coords_2d)

              # data
              img = cv2.imread(self.img_path[i], cv2.IMREAD_COLOR)
              b, g, r = cv2.split(img)
              img = cv2.merge([r, g, b])
              self.x_data.append(img)

              # fipped data
              img = cv2.flip(img, 1)
              self.x_data.append(img)

        elif method == 'test':
            self.root = self.root + 'test/'
            self.img_path = sorted(glob(self.root + 'rgb/*.jpg'))

            for i in tqdm.tqdm(range(len(self.img_path))):
              img = cv2.imread(self.img_path[i], cv2.IMREAD_COLOR)
              b, g, r = cv2.split(img)
              img = cv2.merge([r, g, b])
              self.x_data.append(img)

              num = self.img_path[i].split('.')[0].split('/')[-1]
              img_pkl = self.root + 'meta/' + str(num) + '.pkl'
              pkl = pandas.read_pickle(img_pkl)
              coords_2d = pkl['coords_2d']
              self.y_data.append(coords_2d)

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        transform1 = torchvision.transforms.ToTensor()
        new_x_data = transform1(self.x_data[idx])
        return new_x_data, self.y_data[idx]


class Trainer(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self._build_model()

        dataset = ObmanDataset(method='train')
        self.root = dataset.root
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

#         # Load of pretrained_weight file
#         weight_root = self.root.split('/')
#         del weight_root[-2]
#         weight_root = "/".join(weight_root)
#         weight_PATH = weight_root + 'pretrained_weight.pth'

#         # LOAD PARAMETERS INTO MODEL
#         self.poseNet.load_state_dict(torch.load(weight_PATH))

    def _build_model(self):
        # 2d pose estimator
        poseNet = CPM2DPose()
        self.poseNet = poseNet.to(device)
        self.poseNet.train()

        print('Finish build model.')

    def skeleton2heatmap(self, _heatmap, keypoint_targets):
        heatmap_gt = torch.zeros_like(_heatmap, device=_heatmap.device)

        keypoint_targets = (((keypoint_targets)) // 8)
        for i in range(keypoint_targets.shape[0]):
            for j in range(21):
                x = int(keypoint_targets[i, j, 0])
                y = int(keypoint_targets[i, j, 1])
                heatmap_gt[i, j, x, y] = 1

        heatmap_gt = heatmap_gt.detach().cpu().numpy()
        for i in range(keypoint_targets.shape[0]):
            for j in range(21):
                heatmap_gt[i, j, :, :] = cv2.GaussianBlur(heatmap_gt[i, j, :, :], ksize=(3, 3), sigmaX=2, sigmaY=2) * 9 / 1.1772
        heatmap_gt = torch.FloatTensor(heatmap_gt).to(device)
        return heatmap_gt

    def train(self, epochs, learning_rate, tester):

        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.poseNet.parameters(), lr=learning_rate)
        losses = []

        # Loop through epochs
        for epoch in tqdm.tqdm(range(epochs)):
            loss_epoch = 0
            batch_num = 0
            for batch_idx, samples in enumerate(self.dataloader):
                x_train, y_train = samples
                heatmapsPoseNet = self.poseNet(x_train.cuda())
                gt_heatmap = self.skeleton2heatmap(heatmapsPoseNet, y_train)

                # Training happens here
                optimizer.zero_grad()
                loss = loss_func(heatmapsPoseNet, gt_heatmap)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                batch_num += 1
                ## Write train result
                if batch_idx % 20 == 0:
                    print('Epoch {:4d}/{} Batch {}/{}'.format(epoch+1, epochs, batch_idx, len(self.dataloader)))

            losses.append(loss_epoch/batch_num)
            print(' Training Set Loss : ', loss_epoch/batch_num)
            print(' Test(Validation) Set Loss : ', tester.test_for_train(self.poseNet))

        # Plot the losses as a graph; this graph can be used to tune the learning rate and the # of epochs.
        plt.plot(losses)

        torch.save(self.poseNet.state_dict(), 'finetunedweight.pth')

        print('Finish training.')

    def exp(self):
        for batch_idx, samples in enumerate(self.dataloader):
            x_train, y_train = samples
            heatmapsPoseNet = self.poseNet(x_train.cuda())
            gt_heatmap = self.skeleton2heatmap(heatmapsPoseNet, y_train)

            print(heatmapsPoseNet.shape)
            print(gt_heatmap.shape)

            if batch_idx == 0:
              break

class Tester(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self._build_model()

        dataset = ObmanDataset(method='test')
        self.root = dataset.root
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.datalen = dataset.__len__()
        self.mse_all_img = []

        # Load finetunedweight.pth file to model
        # self.poseNet.load_state_dict(torch.load('finetunedweight.pth'))

    def _build_model(self):
        # 2d pose estimator
        poseNet = CPM2DPose()
        self.poseNet = poseNet.to(device)
        # Issue 1 : the error explode if it's turned into evaluation mode due to drop out layer
        # self.poseNet.eval()

        print('Finish build model.')

    def heatmap2skeleton(self, heatmapsPoseNet):
        skeletons = np.zeros((heatmapsPoseNet.shape[0], heatmapsPoseNet.shape[1], 2))
        for m in range(heatmapsPoseNet.shape[0]):
            for i in range(heatmapsPoseNet.shape[1]):
                u, v = np.unravel_index(np.argmax(heatmapsPoseNet[m][i]), (32, 32))
                skeletons[m, i, 0] = u * 8
                skeletons[m, i, 1] = v * 8
        return skeletons

    def test(self):
        total_error = 0
        
        for batch_idx, samples in enumerate(self.dataloader):
            x_test, y_test = samples
            heatmapsPoseNet = self.poseNet(x_test.cuda()).cpu().detach().numpy()
            skeletons_in = self.heatmap2skeleton(heatmapsPoseNet)
            
            # Accumulate errors for each skeleton data
            for i in range(skeletons_in.shape[0]):
              total_error += self.calc_error(skeletons_in[i], y_test[i].numpy())

        print("Average Error : ", total_error / 500)

    def test_for_train(self, poseNet_train):
        # Issue 2 : find another way rather than copy the network -> inefficient
        # Should be fixed to be done under evaluation mode
        total_error = 0
        
        for batch_idx, samples in enumerate(self.dataloader):
            x_test, y_test = samples
            heatmapsPoseNet = poseNet_train(x_test.cuda()).cpu().detach().numpy()
            skeletons_in = self.heatmap2skeleton(heatmapsPoseNet)
            
            # Accumulate errors for each skeleton data
            for i in range(skeletons_in.shape[0]):
              total_error += self.calc_error(skeletons_in[i], y_test[i].numpy())

        return total_error / 500

    # Calculates the error for one predicted skeleton and one ground truth skeleton
    def calc_error(self, pred, gt):
        err = 0

        for i in range(num_joints):
          err += np.linalg.norm(pred[i]-gt[i])

        err = err / num_joints

        return err

    # Produce heatmap for each 21 joints for 106.jpg
    def visualize_heatmap_for_106(self):
        for batch_idx, samples in enumerate(self.dataloader):
            x_test, y_test = samples
            heatmapsPoseNet = self.poseNet(x_test.cuda()).cpu().detach().numpy()

            for i in range(21):
                plt.imshow(heatmapsPoseNet[9][i], cmap='hot', interpolation='nearest')
                plt.show()
            
            if batch_idx == 0:
                break

    # Return image, and the corresponding predicted skeleton by the index of the data
    def getskeleton(self, idx):
        for batch_idx, samples in enumerate(self.dataloader):
            if idx < self.batch_size:
                x_test, y_test = samples
                heatmapsPoseNet = self.poseNet(x_test.cuda()).cpu().detach().numpy()
                skeletons_in = self.heatmap2skeleton(heatmapsPoseNet)
                return x_test[idx], skeletons_in[idx]
            else:
              idx -= self.batch_size

def main():

  batchSize = 32
  epochs = 50
  learningRate = 1

  trainer = Trainer(batchSize)
  tester = Tester(batchSize)
  trainer.train(epochs, learningRate, tester)
#   tester.test()

if __name__ == '__main__':
    main()