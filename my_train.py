#/usr/bin/env python
from __future__ import print_function

import cv2

from model import UNet
import os
from torch.utils import data
import numpy as np
from my_tools import *    #修改dataset 和save_img
from burstloss import BurstLoss
import torch
import argparse
from helpers import get_newest_model, make_compared_im
# from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from radam import RAdam
import time

#使用sklearn随机划分训练集和测试集
from sklearn.model_selection import train_test_split

def main():
    a=time.time()
    print(os.getcwd())
    # os.chdir('./Data/BID')
    # print(os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', metavar='bs', type=int, default=2)
    parser.add_argument('--path', type=str, default='./dataSet')
    parser.add_argument('--results', type=str, default='./results/model')
    parser.add_argument('--nw', type=int, default=0)
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--val_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--lr_decay', type=float, default=0.99997)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--tr', type=float, default=0.2)    #划分测试集占全部的比例

    args = parser.parse_args()
    print(args)

    if not os.path.isdir(args.results): os.makedirs(args.results)

    PATH = args.results
    if not args.resume:
        f = open(PATH + "/param.txt", "a+")
        f.write(str(args))
        f.close()

    # writer = SummaryWriter(PATH + '/runs')

    # CUDA for PyTorch
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(device)

    # Parameters
    params = {'batch_size': args.bs,
              'shuffle': True,
              'num_workers': args.nw}

    # Generators
    #划分数据集
    data_folder_all=os.listdir(args.path)   #列举dataset文件夹中的所有子文件夹，每个子文件为1个sample
    train_folder_list,test_folder_list=train_test_split(data_folder_all,test_size=args.tr,random_state=0)   #测试集比例默认为0.2

    print('Initializing training set')
    training_set = Dataset(args.path,train_folder_list, args.max_images)
    training_generator = data.DataLoader(training_set, **params)

    print('Initializing validation set')
    validation_set = Dataset(args.path,test_folder_list,  args.val_size)

    validation_generator = data.DataLoader(validation_set, **params)

    # Model
    model = UNet(in_channel=3,out_channel=3)
    if args.resume:
        models_path = get_newest_model(PATH)
        print('loading model from ', models_path)
        model.load_state_dict(torch.load(models_path))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)


    model.to(device)

    # Loss + optimizer
    # criterion = BurstLoss()
    criterion=torch.nn.MSELoss()
    optimizer = RAdam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size = 8 // args.bs, gamma = args.lr_decay)
    if args.resume:
        n_iter = np.loadtxt(PATH + '/train.txt', delimiter=',')[:, 0][-1]
    else:
        n_iter = 0

    # Loop over epochs
    for epoch in range(args.epochs):
        train_loss = 0.0

        # Training
        model.train()
        for i, (X_batch, y_labels) in enumerate(training_generator):
            #每个i就是minibatch

            #burst_length = np.random.randint(2,5)
            burst_length = 4
            X_batch = X_batch[:,:burst_length,:,:,:]

            X_batch, y_labels = X_batch.to(device).type(torch.float), y_labels.to(device).type(torch.float)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            pred = model(X_batch)
            loss = criterion(pred, y_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.detach().cpu().numpy()
            # writer.add_scalar('training_loss', loss.item(), n_iter)

            if i % 100 == 0 and i > 0:
                loss_printable = str(np.round(train_loss,2))    #2 保留2位有效数字

                f = open(PATH + "/train.txt", "a+")
                f.write(str(n_iter) + "," + loss_printable + "\n")
                f.close()

                print("training loss ", loss_printable)

                train_loss = 0.0

            #保存数据
            if i % 1000 == 0:
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), os.path.join(PATH,'model_' + str(int(n_iter)) + '.pt'))
                else:
                    torch.save(model.state_dict(), os.path.join(PATH, 'model_' + str(int(n_iter)) + '.pt'))


            if i % 1000 == 0:
                # Validation
                val_loss = 0.0
                with torch.set_grad_enabled(False):
                    model.eval()
                    for v, (X_batch, y_labels) in enumerate(validation_generator):
                        # Alter the burst length for each mini batch

                        burst_length = 4
                        X_batch = X_batch[:, :burst_length, :, :, :]

                        # Transfer to GPU
                        X_batch, y_labels = X_batch.to(device).type(torch.float), y_labels.to(device).type(torch.float)
                        # y_label size (1,3,160,160)

                        # forward + backward + optimize
                        pred = model(X_batch)
                        loss = criterion(pred, y_labels)

                        val_loss += loss.detach().cpu().numpy()

                        if v < 5:
                            im = make_compared_im(pred, X_batch, y_labels)
                            cv2.imwrite('./test.png',im)
                            # writer.add_image('image_' + str(v), im, n_iter)

                    # writer.add_scalar('validation_loss', val_loss, n_iter)
                    loss_printable = str(np.round(val_loss, 4))
                    print('validation loss ', loss_printable)

                    f = open(PATH + "/eval.txt", "a+")
                    f.write(str(n_iter) + "," + loss_printable + "\n")
                    f.close()

            n_iter += args.bs


if __name__ == "__main__":
    main()
