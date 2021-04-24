import os
import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils
# from tensorboardX import SummaryWriter

try:
    from cnn_model.model import Model
    from cnn_model.loss import ssim
    from cnn_model.data import getTrainingTestingData
    from cnn_model.utils import AverageMeter, DepthNorm, colorize
    from cnn_model.constants import MIN_DEPTH, MAX_DEPTH

except:
    from model import Model
    from loss import ssim
    from data import getTrainingTestingData
    from utils import AverageMeter, DepthNorm, colorize
    from constants import MIN_DEPTH, MAX_DEPTH


from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import pathlib
import threading

# PATH = str(pathlib.Path(__file__).parent.absolute()) + "/saved_model"
CUDA = torch.cuda.is_available()

LEARNING_RATE = 0.0001 * 0.01
VAL_RANGE = 1000.0
SAVE_IN_RUN = 0
LOGS_DIR = str(pathlib.Path(__file__).parent.absolute()) + "/logs/"
SSIM_WEIGHT = 1.0
L1_WEIGHT = 0.1


def save_model(model, train_size, lr, epoch, time):
    SAVED = False
    PATH = str(pathlib.Path(__file__).parent.absolute()) + "/saved_model"
    while not SAVED:
        try:

            torch.save(model.state_dict(), os.path.join(PATH, time + f"_{train_size}_{lr}_{epoch}" + ".pth"))
            model_name = time + f"_{train_size}_{lr}_{epoch}" + ".pth"
            log_name = time + "_log.txt"
            SAVED = True
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
        date_time = now.strftime("%H%M%S")
        SAVE_IN_RUN = 1
        print(f"\n[{date_time}] Changed save in run to: {SAVE_IN_RUN}\n")

    print(f"Starting TRAIN - GOOD LUCK")

    threading.Thread(target=save_in_run_thread).start()

    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=LEARNING_RATE, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=4, type=int, help='batch size')
    args = parser.parse_args()

    # Create model with gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'

    if CUDA:
        model = Model().cuda()
        # model = nn.DataParallel(model,device_ids=[0,1,2,3])

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

    # Logging
    try:
        writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)
    except:
        pass

    # Loss
    l1_criterion = nn.L1Loss()
    now = datetime.datetime.now()  # current date and time
    date_time = now.strftime("%H%M%S")
    to_print = f"Learning Rate = {LEARNING_RATE}, val_range = {VAL_RANGE}"
    with open(LOGS_DIR + date_time + "_log.txt", "a") as text_file:
        print(to_print, file=text_file)

    N = len(train_loader)
    # Start training...
    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()

        # Switch to train mode
        model.train()

        end = time.time()

        for i, sample_batched in enumerate(train_loader):
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
            # plt.imshow(image[3].permute(1, 2, 0))
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

            ## trial with sigmoid normalize the output
            m = nn.Sigmoid()
            output = m(output)

            # Compute the loss
            l_depth = l1_criterion(output, depth_n)

            l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range=VAL_RANGE)) * 0.5, 0, 1)

            # with open("loss.txt", "a") as text_file:
            #     print([l_depth.item(),l_ssim.item()], file=text_file)

            loss = (1.0 * l_ssim) + (0.1 * l_depth)

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
    with torch.cuda.device(2):
        main()
