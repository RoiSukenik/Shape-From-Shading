import os
import time
import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

try:
    from cnn_model.model import Model
    from cnn_model.loss import ssim
    from cnn_model.data import getTrainingTestingData
    from cnn_model.utils import AverageMeter, DepthNorm, colorize
    from cnn_model.constants import ZIP_NAME, MIN_DEPTH, MAX_DEPTH, LEARNING_RATE, EPOCHS, VAL_RANGE, LOGS_DIR, \
        SSIM_WEIGHT, L1_WEIGHT, USE_SCHEDULER, SCHEDULER_STEP_SIZE, SCHEDULER_GAMMA
    # from cnn_model.loss_rt_graph import start_graph_thread
    # from cnn_model import loss_rt_graph

except:
    from model import Model
    from loss import ssim
    from data import getTrainingTestingData
    from utils import AverageMeter, DepthNorm, colorize
    from constants import ZIP_NAME, MIN_DEPTH, MAX_DEPTH, LEARNING_RATE, EPOCHS, VAL_RANGE, LOGS_DIR, SSIM_WEIGHT, \
        L1_WEIGHT, USE_SCHEDULER, SCHEDULER_STEP_SIZE, SCHEDULER_GAMMA
    # from loss_rt_graph import start_graph_thread
    # import loss_rt_graph

from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import pathlib
import threading
import random

# PATH = str(pathlib.Path(__file__).parent.absolute()) + "/saved_model"
# CUDA = False
CUDA = torch.cuda.is_available()

SAVE_IN_RUN = 0


def save_model(model, train_size, lr, epoch, time):
    SAVED = False
    PATH = str(pathlib.Path(__file__).parent.absolute()) + "/saved_model"
    while not SAVED:
        try:
            model_name =ZIP_NAME + "_" + time + f"_{train_size}_{format(lr, '.2e')}_{epoch}" + ".pth"
            torch.save(model.state_dict(), os.path.join(PATH,
                                                        model_name))
            log_name = time + "_log.txt"
            SAVED = True

            with open("last_model.txt", "w") as text_file:
                print(model_name, file=text_file)

            print(f"MODEL SAVED SUCSSEFFULLY IN:\n{model_name}")
            print(f"Log name is:\n{log_name}")
        except:
            print(PATH)
            a = input("Change Path? (y/n)")
            if a == 'n':
                continue
            else:
                PATH = input("New Path?")


def main():
    # Arguments
    global SAVE_IN_RUN

    def save_in_run_thread():
        global SAVE_IN_RUN
        to_change = 0
        # while to_change != "y":
        to_change = input("\nStarting Thread - To save model midrun press Enter\n\n")
        now = datetime.datetime.now()  # current date and time
        date_time = now.strftime("%d/%m/%Y_%H%M%S")
        SAVE_IN_RUN = 1
        print(f"\n[{date_time}] Changed save in run to: {SAVE_IN_RUN}\n")

    print(f"Starting TRAIN - GOOD LUCK")

    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=EPOCHS, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=LEARNING_RATE, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=4, type=int, help='batch size')
    args = parser.parse_args()

    # Create model with gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    if CUDA:
        model = Model().cuda()
        model = nn.DataParallel(model,device_ids=[3])
        model.to(f'cuda:{model.device_ids[0]}')
        print('cuda enable.')
    else:
        model = Model()
        print('cpu enable.')

    print('Model created.')

    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    batch_size = 4  # batch size is how many pics at one go
    prefix = 'densenet_' + str(batch_size)

    # Load data
    train_loader, test_loader = getTrainingTestingData(batch_size=batch_size)
    test_loader = 0
    N = len(train_loader)

    ### ADDED RANDOM BATCHING
    train_loader = [sample_batched for sample_batched in train_loader]
    # batch_ind_lst = [i for i in range(len(train_loader))]

    # Logging
    try:
        writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)
    except:
        pass

    # Loss
    l1_criterion = nn.L1Loss()
    now = datetime.datetime.now()  # current date and time
    date_time = now.strftime("%d%m%Y_%H%M%S")
    to_print = f"Learning Rate = {LEARNING_RATE}, val_range = {VAL_RANGE}, SCHEDULER_STEP_SIZE = {SCHEDULER_STEP_SIZE}, SCHEDULER_GAMMA = {SCHEDULER_GAMMA},"
    with open(LOGS_DIR + date_time + "_log.txt", "a") as text_file:
        print(to_print, file=text_file)


    try:
        with open("last_log.txt", "w") as text_file:
            print(date_time , file=text_file)
    except:
        pass

    print("Training Params:\n", to_print)

    if USE_SCHEDULER:
        scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    # Start training...
    threading.Thread(target=save_in_run_thread).start()
    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()

        # Switch to train mode
        model.train()

        end = time.time()
        # random.shuffle(batch_ind_lst)
        batch_ind_lst = np.random.choice(N, N, replace=False)
        # for i, sample_batched in enumerate(train_loader):
        # WRONG IND 88
        for i, rand_batch_ind in enumerate(batch_ind_lst):
            sample_batched = train_loader[rand_batch_ind]
            optimizer.zero_grad()

            # Prepare sample and target
            if CUDA:
                image = torch.autograd.Variable(sample_batched['image'].cuda())
                depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
            else:
                image = torch.autograd.Variable(sample_batched['image'])
                depth = torch.autograd.Variable(sample_batched['depth'])


            # Normalize depth
            depth_n = DepthNorm(depth)

            # Fix for one output
            depth_n = depth_n[0]
            depth_n = depth_n.unsqueeze(0)
            depth_n = depth_n[0][0]
            depth_n = torch.reshape(depth_n, (1, 1, depth_n.shape[0], depth_n.shape[1]))

            ### debug show input img
            # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            # fig.suptitle(f'rand_batch_ind - {rand_batch_ind}')
            # ax1.imshow(image[0].cpu().permute(1, 2, 0))
            # ax2.imshow(image[1].cpu().permute(1, 2, 0))
            # ax3.imshow(image[2].cpu().permute(1, 2, 0))
            # ax4.imshow(image[3].cpu().permute(1, 2, 0))
            # plt.show()

            # plt.imshow(depth[3].permute(1, 2, 0))
            # plt.show()


            # Predict
            output = model(image)
            if SAVE_IN_RUN:
                print("Saving Model:")
                save_model(model, N, LEARNING_RATE, epoch, date_time)
                SAVE_IN_RUN = 0
                threading.Thread(target=save_in_run_thread).start()

            # frequency = depth_n==1
            # one_occour = frequency.sum().item()
            # num_of_elements = torch.numel(depth_n)
            # total_per = (one_occour /num_of_elements) * 100


            # MASK ATTEMPT
            background = depth_n != 1

            depth_masked = torch.masked_select(depth_n, background)
            output_masked = torch.masked_select(output, background)

            new_shape = depth_masked.shape[0]

            depth_masked = torch.reshape(depth_masked, (1, 1, 1, new_shape))
            output_masked = torch.reshape(output_masked, (1, 1, 1, new_shape))

            ## trial with sigmoid normalize the output
            m = nn.Sigmoid()
            output_masked = m(output_masked)

            # depth_n[background] = 0
            # output_masked = output
            # output_masked[background] = 0
            # output[background] = 0

            # Compute the loss
            l_depth = l1_criterion(output_masked, depth_masked)
            l_ssim = torch.clamp((1 - ssim(output_masked, depth_masked, val_range=VAL_RANGE)) * 0.5, 0, 1)

            loss = (SSIM_WEIGHT * l_ssim) + (L1_WEIGHT * l_depth)

            # Update step
            losses.update(loss.data.item(), image.size(0))
            loss.backward()
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val * (N - i))))

            # Log progress
            niter = epoch * N + i
            if i % 10 == 0:
                # Print to console
                now = datetime.datetime.now()

                current_time = now.strftime("%H:%M:%S")

                # print('Epoch: [{0}][{1}/{2}]\t'
                #       'Curr Time {current_time}\t'
                #       'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                #       'ETA {eta}\t'
                #       'Loss {loss.val:.4f} ({loss.avg:.4f})'
                #       .format(epoch, i, N,current_time=current_time, batch_time=batch_time, loss=losses, eta=eta))
                to_print = 'Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:.3f} ({batch_time.sum:.3f})\tETA {eta}\tLoss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, i, N, batch_time=batch_time, loss=losses, eta=eta)
                print(to_print)
                with open(LOGS_DIR + date_time + "_log.txt", "a") as text_file:
                    print(to_print, file=text_file)

                to_print2 = 'Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:.3f} ({batch_time.sum:.3f})\tETA {eta}\tLoss {loss[0]:.4f} ({loss[1]:.4f})'.format(
                    epoch, i, N, batch_time=batch_time, loss=[l_depth.item(), l_ssim.item()], eta=eta)

                with open(LOGS_DIR + date_time + "_loss_log.txt", "a") as text_file:
                    print(to_print2, file=text_file)

                # Log to tensorboard
                try:
                    writer.add_scalar('Train/Loss', losses.val, niter)

                except:
                    pass

            try:
                if i % 300 == 0:
                    LogProgress(model, writer, test_loader, niter)

            except:
                pass

        # Record epoch's intermediate results
        if USE_SCHEDULER:
            scheduler.step()

        try:
            LogProgress(model, writer, test_loader, niter)

        except:
            pass
        try:
            writer.add_scalar('Train/Loss.avg', losses.avg, epoch)

        except:
            pass
    epoch = args.epochs
    save_model(model, N, LEARNING_RATE, epoch, date_time)


try:
    def LogProgress(model, writer, test_loader, epoch):
        model.eval()
        sequential = test_loader
        sample_batched = next(iter(sequential))

        if CUDA:
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
        else:
            image = torch.autograd.Variable(sample_batched['image'])
            depth = torch.autograd.Variable(sample_batched['depth'])

        if epoch == 0: writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
        if epoch == 0: writer.add_image('Train.2.Depth',
                                        colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)),
                                        epoch)
        output = DepthNorm(model(image))
        writer.add_image('Train.3.Ours', colorize(vutils.make_grid(output.data, nrow=6, normalize=False)), epoch)
        writer.add_image('Train.3.Diff',
                         colorize(vutils.make_grid(torch.abs(output - depth).data, nrow=6, normalize=False)), epoch)
        del image
        del depth
        del output

except:
    pass

if __name__ == '__main__':
    with torch.cuda.device(3):
        main()
