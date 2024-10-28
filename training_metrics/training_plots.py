# %%
import numpy as np
import matplotlib.pyplot as plt


# LOG_PATH = '/imec/other/dl4ms/nicule52/work/radarswin/radarswin_checkpoints/g_bbox_no_merge_best/log_rank0.txt'
# LOG_PATH = '/imec/other/dl4ms/nicule52/work/radarswin/radarswin_checkpoints/g_big400/log_rank0.txt'
LOG_PATH = '../model_code/radarswin_tiny/default/log_rank0.txt'

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
    val_loss = [np.mean(val_loss[i:i+3]) for i in range(0, len(val_loss), 3)]
    # val_loss = [val_loss[i+1] for i in range(0, len(val_loss), 3)]

    plt.plot(np.arange(len(train_loss)) / 21, train_loss, label='train')
    plt.plot(np.arange(len(val_loss)) * len(train_loss) / len(val_loss) / 21, val_loss, label='val')
    print(len(train_loss))
    print(len(val_loss))
    # plt.xlim(0, 455)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.ylim(1, 3)
    plt.grid()

# %%
# with open('/imec/other/dl4ms/nicule52/work/radarswin/radar-swin/radarswin_tiny/default/log_rank0.txt', 'r') as f:
with open('../model_code/radarswin_tiny/default/log_rank0.txt', 'r') as f:
    lines = f.readlines()

    average_ap = []
    ap_thrs = [0.5, 1, 2, 4]
    aps = {'0.5' : [], '1' : [], '2' : [], '4' : []}
    for line in lines:
        print(line)
        if 'AP' in line:
            average_ap.append(float((line.split('AP ')[1].split()[0])))
            aps['0.5'].append(float((line.split('AP ')[1].split()[1])[2:]))
            aps['1'].append(float((line.split('AP ')[1].split()[2])))
            aps['2'].append(float((line.split('AP ')[1].split()[3])))
            aps['4'].append(float((line.split('AP ')[1].split()[4])[:-2]))

    x_axis = np.arange(len(average_ap))

    # average_ap = np.convolve(average_ap, np.ones(20)/21, mode='full')
    # average_ap = average_ap[:-19]

    plt.figure()
    plt.plot(x_axis, average_ap, label='AP average')
    plt.xlabel('Epoch')
    plt.ylabel('AP (average precision)')
    plt.grid()
    plt.ylim(0, 0.5)
    # plt.xlim(0, 455)
    for ap_thr in ap_thrs:
        plt.plot(x_axis, aps[str(ap_thr)], label=f'AP {ap_thr}', linestyle='--')

    plt.legend()
    plt.show()

print("gogu")

# %%
