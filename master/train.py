#/usr/bin/env python
from __future__ import print_function
from model import UNet
import os
from torch.utils import data
import numpy as np
from master.tools import Dataset
from burstloss import BurstLoss
import torch
import argparse
from helpers import get_newest_model, make_im
# from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from radam import RAdam
import time

def main():
    a=time.time()
    print(os.getcwd())
    # os.chdir('./Data/BID')
    # print(os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', metavar='bs', type=int, default=1)
    parser.add_argument('--path', type=str, default='./data')
    parser.add_argument('--train_results', type=str, default='./train_results/model')
    parser.add_argument('--nw', type=int, default=0)
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--val_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--lr_decay', type=float, default=0.99997)
    parser.add_argument('--kernel_lvl', type=float, default=1)
    parser.add_argument('--noise_lvl', type=float, default=1)
    parser.add_argument('--motion_blur', type=bool, default=False)
    parser.add_argument('--homo_align', type=bool, default=False)
    parser.add_argument('--resume', type=bool, default=False)

    args = parser.parse_args()

    print()
    print(args)
    print()

    if not os.path.isdir(args.results): os.makedirs(args.results)

    PATH = args.results
    if not args.resume:
        f = open(PATH + "/param.txt", "a+")
        f.write(str(args))
        f.close()

    # writer = SummaryWriter(PATH + '/runs')

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    # use_cuda=False
    device = torch.device('cuda:0' if use_cuda else "cpu")
    print(device)

    # Parameters
    params = {'batch_size': args.bs,
              'shuffle': True,
              'num_workers': args.nw}

    # Generators
    print('Initializing training set')
    training_set = Dataset(args.path + '/train/', args.max_images,
                           args.kernel_lvl, args.noise_lvl, args.motion_blur, args.homo_align)
    training_generator = data.DataLoader(training_set, **params)

    print('Initializing validation set')
    validation_set = Dataset(args.path + '/test/',  args.val_size,
                             args.kernel_lvl, args.noise_lvl, args.motion_blur, args.homo_align)

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

    ##test gpu use
    print('######################before load model##########################')
    gpu_memory_al=torch.cuda.memory_allocated(device=device)
    print('the gpu memory allocated: {}'.format(gpu_memory_al/1024/1024))

    gpu_memory_al=torch.cuda.memory_reserved(device=device)
    print('the gpu memory reserved: {}'.format(gpu_memory_al/1024/1024))

    print(os.system('nvidia-smi -i 0'))
    model.to(device)

    # Loss + optimizer
    criterion = BurstLoss()
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
            # Alter the burst length for each mini batch

            burst_length = np.random.randint(2,9)
            X_batch = X_batch[:,:burst_length,:,:,:]

            # k=np.array(X_batch[0,0,:,:,:],np.uint8)
            # k=np.transpose(k,[1,2,0])
            # cv2.imshow('test0',k)
            # cv2.waitKey(0)
            #
            # k=np.array(X_batch[0,1,:,:,:],np.uint8)
            # k=np.transpose(k,[1,2,0])
            # cv2.imshow('test1',k)
            # cv2.waitKey(0)
            #
            # k=np.array(X_batch[0,2,:,:,:],np.uint8)
            # k=np.transpose(k,[1,2,0])
            # cv2.imshow('test2',k)
            # cv2.waitKey(0)
            #
            # k=np.array(X_batch[0,3,:,:,:],np.uint8)
            # k=np.transpose(k,[1,2,0])
            # cv2.imshow('test3',k)
            # cv2.waitKey(0)

            # Transfer to GPU
            #y_label是 3*W*H 预测一整张图片
            print('minibatch: {}'.format(i))
            print('######################after data transfer to gpu##########################')
            ##test gpu use
            print('######################after data transfer to gpu##########################')
            gpu_memory_al=torch.cuda.memory_allocated(device=device)
            print('the gpu memory allocated: {}'.format(gpu_memory_al/1024/1024))

            gpu_memory_al=torch.cuda.memory_reserved(device=device)
            print('the gpu memory reserved: {}'.format(gpu_memory_al/1024/1024))

            print(os.system('nvidia-smi -i 0'))

            X_batch, y_labels = X_batch.to(device).type(torch.float), y_labels.to(device).type(torch.float)

            # zero the parameter gradients
            optimizer.zero_grad()

            ##test gpu use
            print('######################after data transfer to gpu##########################')
            gpu_memory_al=torch.cuda.memory_allocated(device=device)
            print('the gpu memory allocated: {}'.format(gpu_memory_al/1024/1024))

            gpu_memory_al=torch.cuda.memory_reserved(device=device)
            print('the gpu memory reserved: {}'.format(gpu_memory_al/1024/1024))

            print(os.system('nvidia-smi -i 0'))

            # forward + backward + optimize
            pred = model(X_batch)
            loss = criterion(pred, y_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.detach().cpu().numpy()
            # writer.add_scalar('training_loss', loss.item(), n_iter)

            if i % 100 == 0 and i > 0:
                loss_printable = str(np.round(train_loss,2))

                f = open(PATH + "/train.txt", "a+")
                f.write(str(n_iter) + "," + loss_printable + "\n")
                f.close()

                print("training loss ", loss_printable)

                train_loss = 0.0

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

                        burst_length = np.random.randint(2, 9)
                        X_batch = X_batch[:, :burst_length, :, :, :]

                        # Transfer to GPU
                        X_batch, y_labels = X_batch.to(device).type(torch.float), y_labels.to(device).type(torch.float)

                        # forward + backward + optimize
                        pred = model(X_batch)
                        loss = criterion(pred, y_labels)

                        val_loss += loss.detach().cpu().numpy()

                        if v < 5:
                            im = make_im(pred, X_batch, y_labels)
                            # writer.add_image('image_' + str(v), im, n_iter)

                    # writer.add_scalar('validation_loss', val_loss, n_iter)

                    loss_printable = str(np.round(val_loss, 2))
                    print('validation loss ', loss_printable)

                    f = open(PATH + "/eval.txt", "a+")
                    f.write(str(n_iter) + "," + loss_printable + "\n")
                    f.close()

            n_iter += args.bs

    b=time.time()
    print(b-a)
if __name__ == "__main__":
    torch.cuda.empty_cache()
    print( torch.cuda.max_memory_allocated())
    main()
