#/usr/bin/env python
from __future__ import print_function


from model import UNet
import os
import numpy as np
from my_tools import *    #修改dataset 和save_img
import torch
import argparse
from helpers import get_newest_model_path, make_compared_im
from sendmail import sendmailtolab
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from radam import RAdam
import time
#使用sklearn随机划分训练集和测试集
from sklearn.model_selection import train_test_split

def load_train_validate_data(data_path,batch_size,num_workers,test_size):
    params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': num_workers}


    data_folder_all=os.listdir(data_path)   #列举dataset文件夹中的所有子文件夹，每个子文件为1个sample
    train_folder_list,test_folder_list=train_test_split(data_folder_all,test_size=test_size,random_state=0)   #测试集比例默认为0.2

    print('Initializing training set')
    training_set = Dataset(data_path,train_folder_list)
    training_generator = data.DataLoader(training_set, **params)

    print('Initializing validation set')
    validation_set = Dataset(data_path,test_folder_list)
    validation_generator = data.DataLoader(validation_set, **params)

    return training_generator,validation_generator

#TODO：model是否需要回传
def train_epoch(training_generator,model,epoch,writer,device,criterion,optimizer,scheduler):
    start=time.time()
    train_loss = 0.0
    #Training
    model.train()
    for i, (X_batch, y_labels, img_names) in enumerate(training_generator):
        #每个i就是minibatch


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
        global n_iter
        n_iter +=  args.bs
        writer.add_scalar('training_loss', loss.item(), n_iter)

    end=time.time()
    f = open(PATH + "/train.txt", "a+")
    f.write(str(n_iter) + "," + str(np.round(train_loss,4)) + "\n")
    f.close()

    print('train loss: %.4f,          train time: %.4f min'%(train_loss,(end-start)/60 ))
    return model

def validate_epoch(validation_generator,model,epoch,writer,device,criterion):
    start=time.time()

    val_loss = 0.0
    with torch.set_grad_enabled(False):
        model.eval()
        for v, (X_batch, y_labels, img_names) in enumerate(validation_generator):
            # Alter the burst length for each mini batch

            # Transfer to GPU
            X_batch, y_labels = X_batch.to(device).type(torch.float), y_labels.to(device).type(torch.float)  # y_label size (1,3,160,160)


            # forward + backward + optimize
            pred = model(X_batch)
            loss = criterion(pred, y_labels)

            val_loss += loss.detach().cpu().numpy()
            #保存对比图片
            if v < 2: #只保存前两个batch
                batch_imgs = make_compared_im(pred, X_batch, y_labels)
                for b,img in enumerate(batch_imgs):
                    #epoch i sample j：第i个epoch中第j个sample
                    cv2.imwrite(PATH+'/validate_output_image/epoch{}_sample{}.png'.format(epoch,v*args.bs+b),img)
                    # writer.add_image('image_' + str(v), im, n_iter)


        writer.add_scalar('validation_loss', val_loss, n_iter)

        end=time.time()
        print('validation loss: %.4f,     validation time: %.4f min'%(val_loss,(end-start)/60) )

        f = open(PATH + "/eval.txt", "a+")
        f.write(str(n_iter) + "," + str(np.round(val_loss,4)) + "\n")
        f.close()

    return val_loss


def create_save_folder(train_result_path):
    #TODO:有bug
    folder_name=''

    k=len(os.listdir(train_result_path)) #已经有几个模型了
    folder_name+=str(k+1)


    now=time.localtime()
    folder_name+='_'+str(now.tm_year)+'-'+str(now.tm_mon)+'-'+str(now.tm_mday)
    folder_name+='_'+'epoch'+str(args.epochs)
    folder_name+='_'+'bs'+str(args.bs)
    folder_name+='_'+'lr'+str(args.lr)
    folder_name+='_'+'lr_decay'+str(args.lr_decay)
    folder_name+='_'+'tr'+str(args.tr)

    if args.resume:
        folder_name+='_'+'resume'

    full_name=train_result_path+'/'+folder_name

    os.mkdir(full_name)

    return full_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', metavar='bs', type=int, default=2)
    parser.add_argument('--path', type=str, default='./dataSet')
    parser.add_argument('--train_results', type=str, default='./train_results')
    parser.add_argument('--nw', type=int, default=0)
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--val_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--lr_decay', type=float, default=0.99997)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--tr', type=float, default=0.2)    #划分测试集占全部的比例
    global args #把参数变为全局
    args = parser.parse_args()

    #创建文件夹，防止报错
    if not os.path.isdir(args.train_results): os.makedirs(args.train_results)
    
    # Model
    model = UNet(in_channel=3,out_channel=3)
    
    global n_iter
    if args.resume:
        newest_path = get_newest_model_path(args.train_results)
        print('loading model from ', newest_path)
        #load model
        model.load_state_dict(torch.load(newest_path+'/model/best_model.pt'))
        #n_iter
        n_iter = np.loadtxt(newest_path + '/train.txt', delimiter=',')[:, 0][-1]
    else:
        n_iter=0
    
      
    #创建文件夹
    global PATH
    PATH = create_save_folder(args.train_results)

    if not os.path.isdir(PATH+'/model'): os.mkdir(PATH+'/model')
    if not os.path.isdir(PATH+'/validate_output_image'): os.mkdir(PATH+'/validate_output_image')

    #保存参数 TODO:有bug
    f = open(PATH + "/param.txt", "a+")
    f.write(str(args))
    f.close()

    writer = SummaryWriter(PATH + '/runs')

    # CUDA for PyTorch
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(device)

    # Generators
    training_generator,validation_generator=load_train_validate_data(args.path,args.bs,args.nw,args.tr)

    

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    model.to(device)

    # Loss + optimizer
    criterion=torch.nn.MSELoss()
    optimizer = RAdam(model.parameters(), lr=args.lr)
    #TODO：这个是什么
    scheduler = StepLR(optimizer, step_size = 8 // args.bs, gamma = args.lr_decay)




    MVL=-1   #minimum validation loss
    # Loop over epochs
    for epoch in range(args.epochs):
        print('############EPOCH %d############'%(epoch))
        #模型训练
        model=train_epoch(training_generator,model,epoch,writer,device,criterion,optimizer,scheduler)

        #验证部分
        val_loss=validate_epoch(validation_generator,model,epoch,writer,device,criterion)
            #只保留validate loss最小的model,当val_loss小于MVL时，is_saved为True

        #模型保存与否的判断
        if MVL < 0 or MVL > val_loss:
            MVL , IS_SAVED , bestepoch = val_loss , True , epoch
        else:
            IS_SAVED=False

        if IS_SAVED: #如果当前MVL最小，则说明model最好，保存
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), PATH+'/model/'+'best_model'+'.pt')
            else:
                torch.save(model.state_dict(), PATH+'/model/'+'best_model'+'.pt')

    str_mail = str(MVL) + 'with epoch' + str(bestepoch)
    sendmailtolab('Network done', str_mail)

if __name__ == "__main__":
    main()
