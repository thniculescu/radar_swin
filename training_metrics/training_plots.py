# %%
import numpy as np
import matplotlib.pyplot as plt


# LOG_PATH = '/imec/other/dl4ms/nicule52/radarswin_tiny/g_bbox_no_merge_best/log_rank0.txt'
LOG_PATH = '/imec/other/dl4ms/nicule52/radarswin_tiny/default/log_rank0.txt'

with open(LOG_PATH, 'r') as f:
    lines = f.readlines()
    lines = lines[:-2]

    train_loss = []
    val_loss = []
    for line in lines:
        if 'INFO Train' in line:
            # print(line)
            train_loss.append(float(line.split('loss ')[1].split()[1][1:-1]))
        if 'INFO Test' in line:
            val_loss.append(float(line.split('Loss ')[1].split()[1][1:-1]))

    #replace every 3 consecutive val_loss with their average:
    # val_loss = [np.mean(val_loss[i:i+3]) for i in range(0, len(val_loss), 3)]
    val_loss = [val_loss[i+1] for i in range(0, len(val_loss), 3)]

    plt.plot(np.arange(len(train_loss)) / 12, train_loss, label='train')
    plt.plot(np.arange(len(val_loss)) * len(train_loss) / len(val_loss) / 12, val_loss, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.ylim(1.5, 3)
    plt.grid()

print("gogu")
