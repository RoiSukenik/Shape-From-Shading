import datetime

import numpy as np
import matplotlib.pyplot as plt
import pathlib

# log_id = "07052021_175438"

with open("last_log.txt", "r") as text_file:
    log_id = text_file.readline().strip()

PATH = str(pathlib.Path(__file__).parent.absolute()) + "/logs/"
diff_loss_file = log_id + "_loss_log.txt"
avg_loss_file = log_id + "_log.txt"

diff_loss_file_PATH = PATH + diff_loss_file
avg_loss_file_PATH = PATH + avg_loss_file
output_path = "graph_results/" + avg_loss_file.split('.')[0]


def get_data(input_path):
    ind = [0]
    Loss_lst = []
    Loss_avg_lst = []
    Epoch_lst = []
    last_epoch = 0
    with open(input_path) as f:
        for line in f:
            if "Learning Rate" in line:
                continue
            ind.append(ind[-1] + 1)
            co_line = line[:]
            Loss = line.split('Loss ')[1]
            curr_loss = float(Loss.split(' ')[0])
            avg_loss = float(Loss.split(' ')[1].replace('(', '').replace(')', ''))
            Loss_lst.append(curr_loss)
            Loss_avg_lst.append(avg_loss)
            try:
                Epoch = int(co_line[8:10])
            except:
                Epoch = int(co_line[8])
            if Epoch_lst:
                if Epoch != last_epoch:
                    Epoch_lst.append(ind[-1])
                    last_epoch = Epoch
            else:
                last_epoch = Epoch
                Epoch_lst.append(ind[-1])

    ind.pop(0)
    return ind, Loss_lst, Loss_avg_lst, Epoch_lst


ind, Loss_lst, Loss_avg_lst, Epoch_lst = get_data(avg_loss_file_PATH)

ind2, l1_loss, ssim_loss, Epoch_lst = get_data(diff_loss_file_PATH)

fig, (ax1, ax2) = plt.subplots(2, sharex=True)

ax1.set_title(f'Loss Graph - {log_id}')

ax1_2 = ax1.twinx()
ax1.plot(ind, Loss_lst, 'g-', label='Loss')
ax1_2.plot(ind, Loss_avg_lst, 'r-', label='Loss_avg')

ax1.set_ylabel('Loss_lst data', color='g')
ax1_2.set_ylabel('Loss_avg_lst data', color='r')

plt_colors = ['c']
# self.plt_colors = ['g', 'r', 'c', 'm', 'y', 'k']

for epoch in Epoch_lst:
    new_c = plt_colors.pop(0)
    plt_colors.append(new_c)
    # ax1.axvline(x=epoch, label='epoch = {}'.format(Epoch_lst.index(epoch)+5), linestyle="dashed", color=new_c)
    ax1.axvline(x=epoch, linestyle="dashed", color=new_c)
ax1_2.axvline(x=epoch, label='epoch', linestyle="dashed", color=new_c)

handles, labels = [], []
for ax in fig.axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
ax1.legend(handles, labels, loc=1, fontsize=8)
# plt.savefig(output_path)
# plt.show()

print("Plot saved successfully at " + output_path)


ax2.set_title('Diff Loss Graph')

ax2_2 = ax2.twinx()
ax2.plot(ind, ssim_loss, 'g-', label='ssim Loss')
ax2_2.plot(ind, l1_loss, 'r-', label='l1_loss')

ax2.set_xlabel('ind - each 2 batches')
ax2.set_ylabel('ssim_loss data', color='g')
ax2_2.set_ylabel('l1_loss data', color='r')

plt_colors = ['c']
# self.plt_colors = ['g', 'r', 'c', 'm', 'y', 'k']

for epoch in Epoch_lst:
    new_c = plt_colors.pop(0)
    plt_colors.append(new_c)
    # ax1.axvline(x=epoch, label='epoch = {}'.format(Epoch_lst.index(epoch)+5), linestyle="dashed", color=new_c)
    ax2.axvline(x=epoch, linestyle="dashed", color=new_c)
ax2_2.axvline(x=epoch, label='epoch', linestyle="dashed", color=new_c)

handles, labels = [], []
for ax in fig.axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
ax2.legend(handles, labels, loc=1, fontsize=8)
plt.savefig(output_path)
plt.show()

# print("Plot saved successfully at " + output_path)