# %%
import numpy as np
import matplotlib.pyplot as plt
import os

# LOG_PATH = '../checkpoints/detec/alpha_small/no_static_only_veh/log_rank0.txt'
# LOG_PATH = '../checkpoints/detec/alpha_small/only_veh/log_rank0.txt'
# LOG_PATH = '../checkpoints/detec/alpha_small/all_targets/log_rank0.txt'
# LOG_PATH = '../checkpoints/track/alpha_small/no_static_only_veh/log_rank0.txt'
# LOG_PATH = '../checkpoints/track/alpha_small/only_veh/log_rank0.txt'
LOG_PATH = '../checkpoints/track/alpha_small/all_targets/log_rank0.txt'

# LOG_PATH = '../model_code/alpha_small/all_targets_no_rl_filter_track/log_rank0.txt'


with open(LOG_PATH, 'r') as f:
    lines = f.readlines()

    train_loss = []
    val_loss = []
    for line in lines:
        if 'INFO Train:' in line:
            # print(line)
            train_loss.append(float(line.split('loss ')[1].split()[1][1:-1]))
        if 'INFO Test' in line:
            # print(line)
            val_loss.append(float(line.split('Loss ')[1].split()[1][1:-1]))

    #replace every 3 consecutive val_loss with their average:
    val_loss = [np.mean(val_loss[i:i+6]) for i in range(0, len(val_loss), 6)]
    # val_loss = [val_loss[i+1] for i in range(0, len(val_loss), 3)]

    plt.plot(np.arange(len(train_loss)) / 21, train_loss, label='train')
    plt.plot(np.arange(len(val_loss)) * len(train_loss) / len(val_loss) / 21, val_loss, label='val')
    # print(len(train_loss))
    # print(len(val_loss))
    # plt.xlim(0, 455)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.ylim(1, 3)
    plt.grid()

with open(LOG_PATH, 'r') as f:
    lines = f.readlines()

    average_ap = []
    ap_thrs = [0.5, 1, 2, 4]
    aps = {'0.5' : [], '1' : [], '2' : [], '4' : []}
    for line in lines:
        # print(line)
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
    plt.plot(x_axis, average_ap, label='AP average', color='black')
    plt.xlabel('Epoch')
    plt.ylabel('AP (average precision)')
    plt.grid()
    plt.ylim(0, 0.5)
    # plt.xlim(0, 455)
    for ap_thr in ap_thrs:
        plt.plot(x_axis, aps[str(ap_thr)], label=f'AP {ap_thr}m', linestyle='--')

    plt.legend()
    plt.show()

    start_time = lines[5].split()[1]
    end_time = lines[-1].split()[1]
    #transform training time to seconds
    start_time = [int(x) for x in start_time.split(':')]
    end_time = [int(x) for x in end_time.split(':')]
    start_time = start_time[0] * 3600 + start_time[1] * 60 + start_time[2]
    end_time = end_time[0] * 3600 + end_time[1] * 60 + end_time[2]
    training_time = end_time - start_time
    training_time += 24 * 3600
    training_time = training_time % (24 * 3600)
    print(f'Training time: {training_time // 3600}h {training_time % 3600 // 60}m {training_time % 60}s')
    #print time per epoch
    print(f'Time per epoch: {training_time // len(average_ap)}s')
    print(f'Average AP: {average_ap[-1]}')
    

# %%
