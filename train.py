
import trimesh
from path import Path
from dataset import ModelNet10Datset
from sample_normalize_points import SamplePoints
import numpy as np
import math
import random
import os
import torch
from glob import glob
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import plotly.graph_objects as go
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pointnet import PointNet
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from predict import PredictAndPlot

if __name__=='__main__':

    path=Path("../ModelNet10")


    train_dataset = ModelNet10Datset(path)
    valid_dataset = ModelNet10Datset(path, valid=True, folder='test')
    classes = {i: cat for cat, i in train_dataset.classes.items()}

    print('Train dataset size: ', len(train_dataset))
    print('Valid dataset size: ', len(valid_dataset))
    print('Number of classes: ', len(train_dataset.classes))
    print('Sample pointcloud shape: ', train_dataset[0]['pointcloud'].size())
    print('Class: ',classes[train_dataset[0]['category']])

    batch_size=64
    epochs=3
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size,drop_last=False,shuffle=True)

    pointnet=PointNet()
    print(pointnet)
    total_params = sum(p.numel() for p in pointnet.parameters())
    print("total parameters: ",total_params)
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epochs=2

    def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.0001):
        criterion = torch.nn.NLLLoss()  # Classification loss
        bs=outputs.size(0)
        id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
        id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
        if outputs.is_cuda:
            id3x3=id3x3.cuda()
            id64x64=id64x64.cuda()
        diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
        diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
        return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)

    def train(model, train_loader, val_loader=None,  epochs=2, save=True):
        for epoch in range(epochs):
            pointnet.train()
            running_loss = 0.0
            mean_correct=[]
            i=0
            for data in tqdm(train_loader,ncols= 100):
                inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                optimizer.zero_grad()
                outputs, m3x3, m64x64 = pointnet(inputs.transpose(1,2))

                loss = pointnetloss(outputs, labels, m3x3, m64x64)
                loss.backward()
                optimizer.step()

                pred_choice=outputs.data.max(1)[1]
                correct=pred_choice.eq(labels.data).cpu().sum()
                mean_correct.append(correct.item()/float(batch_size))
                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:
                    print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f mean accuracy: %f' %
                        (epoch + 1, i + 1, len(train_loader), running_loss / 10, np.mean(mean_correct)))
                    running_loss = 0.0

            pointnet.eval()
            correct = total = 0

            # validation
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, __, __ = pointnet(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            print('Valid accuracy: %d %%' % val_acc)

            # save the model
            if save:
                torch.save(pointnet.state_dict(), "save_"+str(epoch)+".pth")

    train(pointnet, train_loader, valid_loader, 1, save=True)
    k=PredictAndPlot()