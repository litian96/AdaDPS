import re
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.sans-serif'] = "Arial"
from mpl_toolkits.axisartist.axislines import Subplot


def errorfill(x, y, yerr, color=None, alpha_fill=0.15, ax=None, label=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = next(ax._get_lines.prop_cycler)['color']
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, label=label, linewidth=3.0)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill, linewidth=0.0)

def parse_log(file_name):
    rounds = []
    loss = []

    for line in open(file_name, 'r'):

        search_train_loss = re.search(r'iter (.*) loss (.*)', line, re.M | re.I)

        if search_train_loss:
            rounds.append(int(search_train_loss.group(1)))
            loss.append(float(search_train_loss.group(2)))

    return rounds, loss

def parse_log2(file_name):
    rounds = []
    loss = []

    for line in open(file_name, 'r'):

        search_train_loss = re.search(r'iter (.*) loss (.*)', line, re.M | re.I)

        if search_train_loss:
            rounds.append(int(search_train_loss.group(1)))
            loss.append(float(search_train_loss.group(2)))

    mean_loss = np.mean(np.asarray(loss).reshape(5,-1), axis=0)
    std_loss = np.std(np.asarray(loss).reshape(5,-1), axis=0)

    return rounds, mean_loss, std_loss


f = plt.figure(figsize=[11, 4])


labels = ['SGD', 'AdaS', 'DP-SGD', 'AdaDPS']
loss_files = ['results2/sgd_lr0.03',
              'results2/sgd_scale_lr0.003',
              'results2/dp-sgd_clip0.5_lr0.1',
              'results2/dp-sgd_scale_100_clip0.5_lr0.2']
colors=[ "#f4b273", "#0033cc"]

ax = plt.subplot(1, 2, 1)

round, loss, std_loss = parse_log2(loss_files[0])
errorfill(np.arange(0, 40000, 40000/len(loss))/1000, np.asarray(loss), std_loss  / np.sqrt(5), label=labels[0], color=colors[0])
#plt.plot(np.arange(80), np.asarray(loss), linewidth=3.0, label=labels[0], c=colors[0])
round, loss, std_loss = parse_log2(loss_files[1])
errorfill(np.arange(0, 40000, 40000/len(loss))/1000, np.asarray(loss), std_loss  / np.sqrt(5), label=labels[1], color=colors[1])
#plt.ylim(0.48, 0.6)
plt.yscale('log')
plt.title('Non-Private', fontsize=22)
plt.xlabel("# iters (x1k)", fontsize=20)
plt.ylabel('loss', fontsize=22)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

ax.tick_params(color='#dddddd')
ax.spines['bottom'].set_color('#dddddd')
ax.spines['top'].set_color('#dddddd')
ax.spines['right'].set_color('#dddddd')
ax.spines['left'].set_color('#dddddd')

plt.legend(fontsize=20, frameon=False)
plt.tight_layout()



ax = plt.subplot(1, 2, 2)
round2, loss, std_loss = parse_log2(loss_files[2])
errorfill(np.arange(0, 2250, 2250/len(loss))/100, np.asarray(loss), std_loss, label=labels[2], color=colors[0])
round2, loss, std_loss = parse_log2(loss_files[3])
errorfill(np.arange(0, 2250, 2250/len(loss))/100, np.asarray(loss), std_loss, label=labels[3], color=colors[1])
plt.title('Private', fontsize=22)
plt.ylim(0.75, 3.3)
plt.xlabel("# iters (x100)", fontsize=20)
plt.ylabel('loss', fontsize=22)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


ax.tick_params(color='#dddddd')
ax.spines['bottom'].set_color('#dddddd')
ax.spines['top'].set_color('#dddddd')
ax.spines['right'].set_color('#dddddd')
ax.spines['left'].set_color('#dddddd')

plt.subplots_adjust(wspace=3)
plt.legend(fontsize=20, frameon=False)
plt.tight_layout()

f.savefig("toy.pdf")

