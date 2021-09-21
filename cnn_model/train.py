import argparse
import datetime
import os
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils

matplotlib.use('Agg')

# from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

RESUME_RUN = "16052021_145154"

try:
    from cnn_model.model import Model
    from cnn_model.loss import ssim
    from cnn_model.data import getTrainingTestingData
    from cnn_model.utils import AverageMeter, DepthNorm, colorize
    from cnn_model.constants import ZIP_NAME, MIN_DEPTH, MAX_DEPTH, LOGS_DIR, NOTES, HYPER_PARAMS, CUDA, GPU_TO_RUN, \
        NET_MIDSAVE_THREAD, SAVE_MODEL
    from cnn_model.log_visualize import loss_graph

    # from cnn_model.loss_rt_graph import start_graph_thread
    # from cnn_model import loss_rt_graph
    from cnn_model.predict import show_net_output, test_predict
    from cnn_model.pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

except:
    from model import Model
    from loss import ssim
    from data import getTrainingTestingData
    from utils import AverageMeter, DepthNorm, colorize
    from constants import ZIP_NAME, MIN_DEPTH, MAX_DEPTH, NOTES, HYPER_PARAMS, CUDA, GPU_TO_RUN, NET_MIDSAVE_THREAD, \
        SAVE_MODEL
    from log_visualize import loss_graph

    # from loss_rt_graph import start_graph_thread
    # import loss_rt_graph
    from predict import show_net_output, test_predict
    from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import threading

# CUDA = False

PATH = Path(__file__).parent.absolute()
SAVE_IN_RUN = 0  # GLOBAL FOR MIDTRAIN SAVE


def save_model(model, output_path, train_size, lr, epoch, time, optimizer):
    SAVED = False
    script_path = Path(__file__).parent.absolute()
    while not SAVED:
        try:
            model_name = ZIP_NAME + "_" + time + f"_{train_size}_{format(lr, '.2e')}_{epoch}" + ".pth"
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),'optimizer': optimizer.state_dict() }
            torch.save(state, str(output_path / model_name))
            # torch.save(model.state_dict(), str(output_path / model_name))
            log_name = time + "_log.txt"
            SAVED = True
            
            with open(os.path.join(script_path, "last_model.txt"), "w") as text_file:
                print(str(os.path.join(output_path, model_name)), file=text_file)

            print(f"MODEL SAVED SUCSSEFFULLY IN:\n{model_name}")
            print(f"Log name is:\n{log_name}")
        except:
            print(script_path)
            a = input("Change Path? (y/n)")
            if a == 'n':
                continue
            else:
                script_path = input("New Path?")


def make_run_dir(date_time, run_start_time):
    run_dir = PATH / "results" / (ZIP_NAME + '_' + run_start_time) / date_time
    plot_dir = run_dir.parent / "loss_plots"
    predict_dir = run_dir.parent / "predict"
    model_dir = run_dir / "saved_model"
    log_dir = run_dir / "log"
    mid_run_dir = run_dir / "mid_run"
    run_dir.mkdir(parents=True, exist_ok=True)  # make results folder
    model_dir.mkdir(parents=True, exist_ok=True)  # make results folder
    log_dir.mkdir(parents=True, exist_ok=True)  # make results folder
    mid_run_dir.mkdir(parents=True, exist_ok=True)  # make results folder
    plot_dir.mkdir(parents=True, exist_ok=True)  # make results folder
    predict_dir.mkdir(parents=True, exist_ok=True)  # make results folder
    return run_dir, model_dir, log_dir, mid_run_dir


def main(hyper_params_dict, train_loader, test_loader=0, run_start_time=None):
    # Arguments

    global SAVE_IN_RUN

    def save_in_run_thread():
        global SAVE_IN_RUN
        to_change = input("\nStarting Thread - To save model midrun press Enter\n\n")
        now = datetime.datetime.now()  # current date and time
        date_time = now.strftime("%d/%m/%Y_%H%M%S")
        SAVE_IN_RUN = 1
        print(f"\n[{date_time}] Changed save in run to: {SAVE_IN_RUN}\n")

    print(f"Starting TRAIN - GOOD LUCK")

    LEARNING_RATE = hyper_params_dict["LEARNING_RATE"]
    EPOCHS = hyper_params_dict["EPOCHS"]
    SSIM_WEIGHT = hyper_params_dict["SSIM_WEIGHT"]
    L1_WEIGHT = hyper_params_dict["L1_WEIGHT"]
    USE_SCHEDULER = hyper_params_dict["USE_SCHEDULER"]
    SCHEDULER_STEP_SIZE = hyper_params_dict["SCHEDULER_STEP_SIZE"]
    SCHEDULER_GAMMA = hyper_params_dict["SCHEDULER_GAMMA"]
    ACCUMULATION_STEPS = hyper_params_dict["ACCUMULATION_STEPS"]
    ADAPTIVE_LEARNER = hyper_params_dict["ADAPTIVE_LEARNER"]

    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=EPOCHS, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=LEARNING_RATE, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=len(GPU_TO_RUN), type=int, help='batch size')
    args = parser.parse_args()

    # Create model with gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    now = datetime.datetime.now()  # current date and time
    date_time = now.strftime("%d%m%Y_%H%M%S")

    run_dir, model_dir, log_dir, mid_run_dir = make_run_dir(date_time, run_start_time)

    if CUDA:
        model = Model().cuda()
        model = nn.DataParallel(model, device_ids=GPU_TO_RUN, output_device=GPU_TO_RUN[-1])
        print('cuda enable.')
    else:
        model = Model()
        print('cpu enable.')

    print('Model created.')

    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    batch_size = len(GPU_TO_RUN)  # batch size is how many pics at one go
    prefix = 'densenet_' + str(batch_size)

    N = len(train_loader)

    # Loss
    l1_criterion = nn.L1Loss()
    # l1_criterion = nn.MSELoss()

    to_print = str(hyper_params_dict)

    with open(str(log_dir / (date_time + "_log.txt")), "a") as text_file:
        print(to_print, file=text_file)

    try:
        with open("last_log.txt", "w") as text_file:
            print(str(log_dir / date_time), file=text_file)
    except:
        pass

    print("Training Params:\n", to_print)

    if USE_SCHEDULER:
        if ADAPTIVE_LEARNER:
            scheduler = ReduceLROnPlateau(optimizer, 'min')
        else:
            scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
    # Start training...

    if NET_MIDSAVE_THREAD:
        mid_save_thread = threading.Thread(target=save_in_run_thread)
        mid_save_thread.start()
    batch_count = 0

    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()

        # Switch to train mode
        model.train()

        end = time.time()
        # random.shuffle(batch_ind_lst)
        # batch_ind_lst = np.random.choice(N, N, replace=False)
        for i, sample_batched in enumerate(train_loader):

        # for i, rand_batch_ind in enumerate(batch_ind_lst):
            batch_count += 1
            # sample_batched = train_loader[rand_batch_ind]
            #optimizer.zero_grad()

            # Prepare sample and target
            if CUDA:
                image = torch.autograd.Variable(sample_batched['image'].cuda(device = GPU_TO_RUN[-1]))
                depth = torch.autograd.Variable(sample_batched['depth'].cuda(device = GPU_TO_RUN[-1], non_blocking=True))
            else:
                image = torch.autograd.Variable(sample_batched['image'])
                depth = torch.autograd.Variable(sample_batched['depth'])

            # Normalize depth
            depth_ns = DepthNorm(depth)

            # Fix for one output
            depth_lst = []
            for dept in depth_ns:
                depth_n1 = dept[0]
                depth_n2 = depth_n1.unsqueeze(0)
                depth_lst.append(depth_n2)
            depth_n = torch.stack(depth_lst)

            ### debug show input img
            # matplotlib.use('TkAgg')
            #import matplotlib.pyplot as plt

            #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            #fig.suptitle(f"epoch {epoch} ind {i}")
            #ax1.imshow(image[0][0].cpu().permute(1, 2, 0))
            #ax2.imshow(image[0][1].cpu().permute(1, 2, 0))
            #ax3.imshow(image[0][2].cpu().permute(1, 2, 0))
            #ax4.imshow(image[0][3].cpu().permute(1, 2, 0))
            #plt.show()

            # plt.imshow(depth[3].permute(1, 2, 0))
            # plt.show()

            # Predict
            output = model(image)
            if SAVE_IN_RUN:
                print("Saving Model:")
                save_model(model, model_dir, N, LEARNING_RATE, epoch, date_time,optimizer)
                SAVE_IN_RUN = 0
                threading.Thread(target=save_in_run_thread).start()



            # MASK ATTEMPT
            # output = output.to(f'cuda:{0}')
            background = depth_n == 1
            depth_masked = depth_n.masked_fill_(background, 0.0)
            output_masked = output.masked_fill_(background, 0.0)


            ## trial with sigmoid normalize the output
            # m = nn.Sigmoid()
            # output_masked = m(output_masked)

            # Compute the loss
            l_depth = l1_criterion(output_masked, depth_masked)


            # trial new ssim
            l_ssim = 1 - ssim(output_masked, depth_masked, data_range=1, size_average=False,
                              nonnegative_ssim=True)  # (N,)

            loss = (SSIM_WEIGHT * l_ssim) + (L1_WEIGHT * l_depth)

            # Update step
            losses.update(loss.sum().item(), image.size(0))

            #Gradient Accumulation
            loss = (loss.sum()) / ACCUMULATION_STEPS
            loss.backward()
            if (batch_count - 1) % ACCUMULATION_STEPS == 0:
                 optimizer.step()
                 # Reset gradients, for the next accumulated batches
                 optimizer.zero_grad()
            #loss.sum().backward()
            #optimizer.step()

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

                to_print = 'Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:.3f} ({batch_time.sum:.3f})\tETA {eta}\tLoss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, i, N, batch_time=batch_time, loss=losses, eta=eta)
                print(to_print)
                with open(str(log_dir / (date_time + "_log.txt")), "a") as text_file:
                    print(to_print, file=text_file)

                to_print2 = 'Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:.3f} ({batch_time.sum:.3f})\tETA {eta}\tLoss {loss[0]:.4f} ({loss[1]:.4f})'.format(
                    epoch, i, N, batch_time=batch_time, loss=[l_depth.sum().item(), l_ssim.sum().item()], eta=eta)

                with open(str(log_dir / (date_time + "_loss_log.txt")), "a") as text_file:
                    print(to_print2, file=text_file)


        try:
            with torch.no_grad():
                test_loss, test_l1_loss, test_ssim_loss = test_predict(model, epoch, test_loader, mid_run_dir)
                print(f"\ntest loss = {test_loss}, test_l1_loss={test_l1_loss}, test_ssim_loss={test_ssim_loss} \n")
                to_print3 = 'Epoch: [{0}][{1}/{2}]\tTime {batch_time:.3f} ({batch_time:.3f})\tETA {eta}\tLoss {loss[0]:.4f} ({loss[1]:.4f})'.format(
                    epoch, i, N, batch_time=test_loss, loss=[test_l1_loss, test_ssim_loss], eta=eta)
                with open(str(log_dir / (date_time + "_loss_test_log.txt")), "a") as text_file:
                    print(to_print3, file=text_file)

        except Exception as e:
            print(e)

        # Record epoch's intermediate results
        if USE_SCHEDULER:
            if ADAPTIVE_LEARNER:
                scheduler.step(losses.avg)
            else:
                scheduler.step()

    epoch = args.epochs

    with torch.no_grad():
        test_predict(model, epoch, test_loader, run_dir, hyper_params_dict)

    loss_graph(str(log_dir / date_time), hyper_params_dict, output_path2=str(run_dir / date_time))

    if SAVE_MODEL:
        save_model(model, model_dir, N, LEARNING_RATE, epoch, date_time,optimizer)
    if NET_MIDSAVE_THREAD:
        os._exit(1)
    return model


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

def wrapper(gen):
  while True:
    try:
      yield next(gen)
    except StopIteration:
      break
    except Exception as e:
      print(e) # or whatever kind of logging you want

if __name__ == '__main__':
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    # with torch.cuda.device(GPU_TO_RUN):
    train_loader, test_loader = getTrainingTestingData(batch_size=len(GPU_TO_RUN))

    # train_loader = [sample_batched for sample_batched in train_loader]
    # train_loader =list(wrapper(sample_batched for sample_batched in train_loader))

    now = datetime.datetime.now()  # current date and time
    date_time = now.strftime("%d%m%Y_%H%M%S")
    if RESUME_RUN:
        date_time = RESUME_RUN
    for i in range(len(HYPER_PARAMS["LEARNING_RATE"])):
        print(f"\nRun Number: {i}\n")
        curr_hyper_dict = {}
        for key in HYPER_PARAMS:
            curr_hyper_dict[key] = HYPER_PARAMS[key][i]
        model = main(curr_hyper_dict, train_loader, test_loader, date_time)

        del model
        gc.collect()
        torch.cuda.empty_cache()
