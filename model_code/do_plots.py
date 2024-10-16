import matplotlib.pyplot as plt
import numpy as np
import torch

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))


def plot_bboxes(ax, anns, ground_truth):
    if ground_truth is True:
        color = 'r'
        line_label = 'target'
    else:
        color = 'g'
        line_label = 'prediction'

    init_anns = np.hstack((anns, np.radians(np.arange(360).reshape(-1, 1))))
    init_anns = init_anns[init_anns[:, -2] == 1]

    car_ori = np.arcsin(init_anns[:, 2]) + init_anns[:, -1]
    car_size = init_anns[:, 4:6] * init_anns[:, 1].reshape(-1, 1)
    car_range = init_anns[:, 1]

    # print(car_ori)
    # print(car_size)
    # print(car_range)

    #make corners of rectangles car_size[:, 1] long by car_size[:, 0] wide, facing Ox
    #car_size[:, 1] is parallel to Ox
    #car_size[:, 0] is parallel to Oy
    car_corners = np.array([[-car_size[:, 1] / 2, -car_size[:, 0] / 2],
                            [-car_size[:, 1] / 2, car_size[:, 0] / 2],
                            [car_size[:, 1] / 2, car_size[:, 0] / 2],
                            [car_size[:, 1] / 2, -car_size[:, 0] / 2]])
    

    car_corners = np.transpose(car_corners, (2, 0, 1))

    #rotate each box around the origin from car_corners[x, :, :] by car_ori[x]
    car_corners = np.array([np.dot(car_corners[x], np.array([[np.cos(car_ori[x]), np.sin(car_ori[x])], [-np.sin(car_ori[x]), np.cos(car_ori[x])]])) for x in range(car_corners.shape[0])])
    #convert car_range and init_anns[:, -1] from polar coordinates to cartesian coordinates
    car_trans = np.array([np.array([car_range[x] * np.cos(init_anns[x, -1]), car_range[x] * np.sin(init_anns[x, -1])]) for x in range(car_corners.shape[0])])
    

    for x in range(car_corners.shape[0]):
        for y in range(car_corners.shape[1]):
            car_corners[x, y, :] += car_trans[x]


    #convert each pair from car_trans[:, :, 2] to polar
    car_corners = car_corners.reshape(-1, 2)
    car_trans_final = np.array([np.array([np.hypot(car_corners[x, 0], car_corners[x, 1]), np.arctan2(car_corners[x, 1], car_corners[x, 0])]) for x in range(car_corners.shape[0])])
    car_trans_final = car_trans_final.reshape(-1, 4, 2)
    for x in range(car_trans_final.shape[0]):
        gigi = car_trans_final[x]
        gigi = np.vstack((gigi, gigi[0]))
        ax.plot(gigi[:, 1], gigi[:, 0], c=color, label=line_label)

    for x in range(car_range.shape[0]):
        if ground_truth is False:
            color = 'k'
        ax.arrow(init_anns[x, -1], car_range[x], 0, car_range[x] * init_anns[x, 6], lw=4, fc=color, ec=color, head_width=0, head_length=0)
        
        if ground_truth is True:
            ranges = [car_range[x], np.hypot(car_range[x], init_anns[x, 7] * car_range[x])]
            angles = [init_anns[x, -1], init_anns[x, -1] + np.arcsin(init_anns[x, 7] * car_range[x] / ranges[1])]       
            ax.arrow(angles[0], ranges[0], angles[1] - angles[0], ranges[1] - ranges[0], lw=4, fc=color, ec=color, head_width=0, head_length=0)





def do_plots(config, data_loader_val, model):
    model.cuda()

    # take a sample from the validation set
    for i, (samples, target) in enumerate(data_loader_val):

        with torch.amp.autocast('cuda', enabled=config.AMP_ENABLE):
            outputs = model(samples)

        # pred      : B, 360, 8     : hmap, range, orientation [sin, cos], size [w l], velocity [vr vt]
        # target    : B, 360, 8 + 1 : hmap, range, orientation [sin, cos], size [w l], velocity [vr vt], mask

        for j in np.random.randint(50, samples.shape[0] - 1, 8):
            sweeps = samples[j]
            sweeps = sweeps.cpu().numpy().transpose(1, 2, 0)

            anns = target[j]
            anns = anns.cpu().numpy()

            mask = anns[:, -1]

            preds = outputs[j]
            preds = preds.cpu().detach().numpy()
            #make a mask where preds[x, 0] is larger than both neighbours
            peak_preds_mask = np.logical_and(preds[:, 0] > np.roll(preds[:, 0], 1), preds[:, 0] > np.roll(preds[:, 0], -1))
            #make a mask out of preds[:, 0] > 0.25
            preds_mask = np.ma.masked_where(preds[:, 0] >= 0.25, preds[:, 0])
            preds_mask = np.ma.getmask(preds_mask)

            #intersect the two masks
            preds_mask = np.logical_and(preds_mask, peak_preds_mask)

            #add mask to ranges
            preds[~preds_mask, 1] = np.nan

            # transform true to 1 and false to 0 in preds_mask
            preds_mask = preds_mask.astype(int)
            # add preds_mask to preds
            preds = np.hstack((preds, preds_mask.reshape(-1, 1)))


            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 12))
            for idx in range(sweeps.shape[0]):
                ax.scatter(np.radians(np.arange(360)), sweeps[idx, :, 0], marker='o', c='b', s=5, label='radar_sweeps')

            # ax.scatter(np.radians(np.arange(360)), anns[:, 1], marker='x', c='r', s=60)
            ax.set_ylim(0, 60)


            plot_bboxes(ax, preds, ground_truth=False)

            plot_bboxes(ax, anns, ground_truth=True)

            

            #plot predictions + pred_Vr
            # good_preds = np.hstack((preds[~preds_mask], np.radians(np.arange(360)[~preds_mask]).reshape(-1, 1)))
            # print(good_preds.shape)
            # for x in range(good_preds.shape[0]):
            #     ax.arrow(good_preds[x, -1], good_preds[x, 1], 0, good_preds[x, 1] * good_preds[x, 6], lw=4, fc='g', ec='k', head_width=0, head_length=0)

            ax.scatter(np.radians(np.arange(360)), preds[:, 1], marker='o', c='g', s=50, label='target_predicions')
            
                  
            
            legend_without_duplicate_labels(ax)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(np.arange(360), anns[:, 0], c='r')
            ax.plot(np.arange(360), preds[:, 0], c='g')
            #plot horizontal black line at 0.25
            ax.axhline(y=0.25, color='k', linestyle='--', label='selection threshold')
            ax.set_title('heatmap')
            ax.legend()
            ax.set_xlabel('angle bin')

            
            # VR-------------------
            # fig, ax = plt.subplots(figsize=(8, 8))
            # ax.plot(np.arange(360), anns[:, 6], c='r')
            # ax.plot(np.arange(360), preds[:, 6], c='g')
            # ax.set_title('radial velocity')
            # ax.set_xlabel('angle bin')
            # ax.set_ylabel('Vr / predicted range to target')

            # fig, ax = plt.subplots(figsize=(8, 8))
            # ax.plot(np.arange(360), anns[:, 2], c='r')
            # ax.plot(np.arange(360), preds[:, 2], c='g')
            # ax.set_title('orientation, sin(alpha)')
            # ax.set_xlabel('angle bin')
            # ax.set_ylabel('sin(alpha) - relative to angle bin')

            # BBOX SIZE-------------------
            # fig, ax = plt.subplots(figsize=(8, 8))
            # ax.plot(np.arange(360), anns[:, 5], c='b', label='L_t')
            # ax.plot(np.arange(360), preds[:, 5], c='k', label='L_p')
            # ax.plot(np.arange(360), anns[:, 4], c='r', label='W_t')
            # ax.plot(np.arange(360), preds[:, 4], c='g', label='W_p')
            # ax.set_title('bbox W, L')
            # ax.set_xlabel('angle bin')
            # ax.set_ylabel('W, L / predicted range to target')
            # ax.legend()



            # fig, ax = plt.subplots(figsize=(8, 8))
            # ax.plot(np.arange(360), anns[:, 7], c='r')
            # ax.plot(np.arange(360), preds[:, 7], c='g')
            # ax.set_title('tangential velocity')
            # ax.set_xlabel('angle bin')
            # ax.set_ylabel('Vt / predicted range to target')

        if i == 1:
            break

    pass