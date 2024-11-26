import time
import matplotlib.pyplot as plt
import numpy as np
import torch

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))


def get_car_corners(car_size, car_range, car_ori, car_rot):
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
    car_trans = np.array([np.array([car_range[x] * np.cos(car_rot[x]), car_range[x] * np.sin(car_rot[x])]) for x in range(car_corners.shape[0])])
    

    for x in range(car_corners.shape[0]):
        for y in range(car_corners.shape[1]):
            car_corners[x, y, :] += car_trans[x]


    #convert each pair from car_trans[:, :, 2] to polar
    car_corners = car_corners.reshape(-1, 2)
    car_trans_final = np.array([np.array([np.hypot(car_corners[x, 0], car_corners[x, 1]), np.arctan2(car_corners[x, 1], car_corners[x, 0])]) for x in range(car_corners.shape[0])])
    car_trans_final = car_trans_final.reshape(-1, 4, 2)
    return car_trans_final


def plot_bboxes(ax, anns, ground_truth, tracking = False):
    if ground_truth is True:
        color = 'r'
        line_label = 'target'
    else:
        color = 'g'
        line_label = 'prediction'

    # init_anns = np.hstack((anns, np.radians(np.arange(360).reshape(-1, 1))))
    init_anns = np.hstack((anns, np.linspace(0, 2 * np.pi, 360).reshape(-1, 1)))

    init_anns = init_anns[init_anns[:, -2] == 1]

    if ground_truth is True:
        car_ori = np.atan2(init_anns[:, 2], init_anns[:, 3]) + init_anns[:, -1]
    else:
        car_ori = np.atan2(init_anns[:, 2], init_anns[:, 3]) + init_anns[:, -1]
        # car_ori = np.arcsin(init_anns[:, 2]) + init_anns[:, -1]
        # car_ori = np.pi - np.arcsin(init_anns[:, 2]) + init_anns[:, -1]

    car_size = init_anns[:, 4:6] * init_anns[:, 1].reshape(-1, 1)
    car_range = init_anns[:, 1]
    car_rot = init_anns[:, -1]

    # print(car_ori)
    # print(car_size)
    # print(car_range)

    car_corners = get_car_corners(car_size, car_range, car_ori, car_rot)

    
    for x in range(car_corners.shape[0]):
        gigi = car_corners[x]
        gigi = np.vstack((gigi, gigi[0]))
        ax.plot(gigi[:, 1], gigi[:, 0], c=color, label=line_label)

    for x in range(car_range.shape[0]):
        if ground_truth is False:
            color = 'k'

        ax.arrow(init_anns[x, -1], car_range[x], 0, car_range[x] * init_anns[x, 6], lw=4, fc=color, ec=color, head_width=0, head_length=0, alpha=0.5)
        
        if ground_truth is True or tracking is True:
            ranges = [car_range[x], np.hypot(car_range[x], init_anns[x, 7] * car_range[x])]
            angles = [init_anns[x, -1], init_anns[x, -1] + np.arcsin(init_anns[x, 7] * car_range[x] / ranges[1])]       
            ax.arrow(angles[0], ranges[0], angles[1] - angles[0], ranges[1] - ranges[0], lw=4, fc=color, ec=color, head_width=0, head_length=0, alpha=0.5)


def gen_heatmap(anns):
    #range, angle, orientation, size [w l], velocity [vr vt]

    car_ori = anns[:, 2] + anns[:, 1]
    car_size = anns[:, 3:5] * anns[:, 0].reshape(-1, 1)
    car_range = anns[:, 0]
    car_rot = anns[:, 1]

    car_corners = get_car_corners(car_size, car_range, car_ori, car_rot)
    # correction for discontinuity PI/-PI
    hmap = np.zeros((360, ))
    for relevant_corners in car_corners:
        if np.any(relevant_corners[:, 1] > np.pi / 2) and np.any(relevant_corners[:, 1] < -np.pi / 2):
            relevant_corners[:, 1] += 2 * np.pi

        #keep minimum and maximum relevant th
        min_th = int(np.round(np.degrees(np.min(relevant_corners[:, 1]))))
        max_th = int(np.round(np.degrees(np.max(relevant_corners[:, 1]))))

        # relevant_ths = np.array([relevant_ths[min_th], relevant_ths[max_th]])
        # relevant_rs = np.array([relevant_rs[min_th], relevant_rs[max_th]])
        mid = (min_th + max_th) // 2
        std = (max_th - min_th) / 6

        #Max gaussian size = 11degrees
        if max_th - min_th >= 13:
            max_th = mid + 5
            min_th = mid - 5
        
        for i in range(min_th, max_th):
            poz = (i + 180) % 360
            val_gauss = np.exp(-((i - mid) ** 2) / (2 * std ** 2 + 0.001))
            hmap[poz] = max(hmap[poz], val_gauss)
    
    hmap = np.roll(hmap, 180)
    return hmap

def do_plots(config, data_loader_val, model):
    start_time = time.time()
    gigel = np.empty((0, 8))
    model.cuda()
    model.eval()
    keep_next = None

    final_preds = None

    for batch_nr, (samples, target, test_case) in enumerate(data_loader_val):
        test_case = [(test_case[0][i], int(test_case[1][i])) for i in range(len(test_case[0]))]

        if final_preds is None:
            final_preds = {}
            for x in test_case:
                #initial empty list for prediction at sweep 0
                final_preds[x[0]] = {"preds": [np.empty((0, 8))], "input_hmap": [[]]}
                

        if keep_next is not None:
            samples = torch.vstack((keep_next[0], samples))
            target = torch.vstack((keep_next[1], target))
            test_case = keep_next[2] + test_case
            keep_next = None    
        
        sweep_nrs = np.array([x[1] for x in test_case])
        if np.unique(sweep_nrs).shape[0] != 1:
            assert (np.unique(sweep_nrs).shape[0] == 2)
            #some sweeps have a higher number than the previous one
            #keep the higher sweeps for next sample
            next_sweep_id = np.max(sweep_nrs)
            #how many samples have the next_sweep_id
            next_sweep_id_count = np.sum(sweep_nrs == next_sweep_id)
            keep_next = (samples[-next_sweep_id_count:], target[-next_sweep_id_count:], test_case[-next_sweep_id_count:])
            samples = samples[:-next_sweep_id_count]
            target = target[:-next_sweep_id_count]
            test_case = test_case[:-next_sweep_id_count]

        for idx, scene_sweep in enumerate(test_case):
            if batch_nr == 0:
                break
            hmap_pred = gen_heatmap(final_preds[scene_sweep[0]]['preds'][scene_sweep[1] - 1])

            # if scene_sweep[0] == '0006' and scene_sweep[1] < 11:
            #     plt.figure(figsize=(12, 3))
            #     plt.title(f"scene: {scene_sweep[0]}, sweep: {scene_sweep[1] - 1}")
            #     plt.plot(np.arange(360), hmap_pred, 'g')
            #     plt.plot(np.arange(360), samples[idx].cpu().numpy()[3, 0, :], 'r--')
            
            #make a copy 6 times of hmap_pred
            hmap_pred = np.stack([hmap_pred for _ in range(6)], axis=0) #6 x 360
            samples[idx, 3] = torch.tensor(hmap_pred).to(torch.float16) #4th channel = heatmap (before: r, vr, rcs)
            
            
        with torch.amp.autocast('cuda', enabled=config.AMP_ENABLE):
            with torch.no_grad():
                outputs = model(samples)

        outputs = outputs.cpu().numpy()
        print(samples.shape)
        for idx, x in enumerate(outputs):
            #mask x where it's bigger than both neighbours
            peak_preds_mask = np.logical_and(x[:, 0] > np.roll(x[:, 0], 1), x[:, 0] > np.roll(x[:, 0], -1))
            #mask x where x[:, 0] is smaller than 0.25
            preds_mask = np.ma.masked_where(x[:, 0] >= config.CENTERNET.PRED_HEATMAP_THR + 0.15, x[:, 0])
            preds_mask = np.ma.getmask(preds_mask)
            #intersect the two masks
            preds_mask = np.logical_and(preds_mask, peak_preds_mask)
            #keep only unmasked lines of x
            #select indices where preds_mask is False
            x = np.hstack((x, np.linspace(0, 2 * np.pi, config.DATA.INPUT_SIZE[1]).reshape(-1, 1)))


            # x[:, 2] = np.arcsin(x[:, 2])
            x[:, 2] = np.atan2(x[:, 2], x[:, 3]) ###TODO ATAN2 change instead of arcsin

            x = x[preds_mask]
            # hmap, range, orientation [sin, cos], size [w l], velocity [vr vt]
            x = x[:, [1, -1, 2, 4, 5, 6, 7, 0]]
            #range, angle, orientation, size [w l], velocity [vr vt] hmap
            
            #remove clutter big buses in front
            # for idxx in range(x.shape[0]):
            #     if x[idxx, 4] * x[idxx, 0] > 5:
            #         x[idxx, 4] = 5 / x[idxx, 0]
            

            # print("x shape: ", x.shape)
            final_preds[test_case[idx][0]]["preds"].append(np.array(x))
            final_preds[test_case[idx][0]]["input_hmap"].append(samples[idx].cpu().numpy()[3, 0, :])

            
        print("batch: ", batch_nr, "len: ", outputs.shape[0])
    
    np.save('./predictions.npy', final_preds, allow_pickle=True)
    print("time elapsed: ", time.time() - start_time)


def do_plots2(config, data_loader_val, model):
    model.cuda()
    model.eval()

    # take a sample from the validation set
    total = 0
    for i, (samples, target, test_case) in enumerate(data_loader_val):
        if i < 3 or i > 6:
            continue

        test_case = [(test_case[0][i], int(test_case[1][i])) for i in range(len(test_case[0]))]

        print("validation samples shape: ", samples.shape)
        # total += samples.shape[0]
        # print("total samples: ", total)
        # continue

        with torch.amp.autocast('cuda', enabled=config.AMP_ENABLE):
            outputs = model(samples)

        # pred      : B, 360, 8     : hmap, range, orientation [sin, cos], size [w l], velocity [vr vt]
        # target    : B, 360, 8 + 1 : hmap, range, orientation [sin, cos], size [w l], velocity [vr vt], mask
        # for j in range(38, 42):
        # for j in np.random.randint(15, 80, 8):
        # for j in np.random.randint(0, 127, 5):
        # for j in np.random.randint(14, 79, 8):
        for j in range(0, 5):
            sweeps = samples[j]
            sweeps = sweeps.cpu().numpy().transpose(1, 2, 0)

            anns = target[j]
            anns = anns.cpu().numpy()

            # if not np.any(anns[:, 7] > 1):
            #     continue

            mask = anns[:, -1]

            sum_mask = np.sum(mask)
            print("sum mask", sum_mask)
            # if sum_mask == 0:
            #     continue


            preds = outputs[j]
            preds = preds.cpu().detach().numpy()
            #make a mask where preds[x, 0] is larger than both neighbours
            peak_preds_mask = np.logical_and(preds[:, 0] > np.roll(preds[:, 0], 1), preds[:, 0] > np.roll(preds[:, 0], -1))
            
            #RESET MASK
            # peak_preds_mask = np.full_like(peak_preds_mask, True)
            
            #make a mask out of preds[:, 0] > 0.25
            preds_mask = np.ma.masked_where(preds[:, 0] >= config.CENTERNET.PRED_HEATMAP_THR, preds[:, 0])
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
            colors = ['#00FF00', '#00D815', '#00B13C', '#008A63', '#00638A', '#003CB1', '#0015D8', '#0000FF'][::-1]

            for idx in range(sweeps.shape[0]):
                ax.scatter(np.radians(np.arange(360)), sweeps[idx, :, 0],
                        marker='o', s=7, c=colors[idx], 
                   alpha=1 - 0.9 * idx / sweeps.shape[0], label='radar_sweeps')

            # ax.scatter(np.radians(np.arange(360)), anns[:, 1], marker='x', c='r', s=60)
            ax.set_ylim(0, 50)


            plot_bboxes(ax, preds, ground_truth=False, tracking=config.MODEL.TRACKING)

            plot_bboxes(ax, anns, ground_truth=True, tracking=config.MODEL.TRACKING)
            ax.set_title(f'Scene: {test_case[j][0]}, Sweep: {test_case[j][1]}')
            

            #plot predictions + pred_Vr
            # good_preds = np.hstack((preds[~preds_mask], np.radians(np.arange(360)[~preds_mask]).reshape(-1, 1)))
            # print(good_preds.shape)
            # for x in range(good_preds.shape[0]):
            #     ax.arrow(good_preds[x, -1], good_preds[x, 1], 0, good_preds[x, 1] * good_preds[x, 6], lw=4, fc='g', ec='k', head_width=0, head_length=0)

            ax.scatter(np.linspace(0, 2 * np.pi, config.DATA.INPUT_SIZE[1], endpoint=False), preds[:, 1], marker='o', c='g', s=50, label='target_predicions')
            
                  
            
            legend_without_duplicate_labels(ax)
            
            fig, ax = plt.subplots(figsize=(12, 3))
            # fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(np.arange(360), anns[:, 0], c='r')
            ax.plot(np.arange(360), preds[:, 0], c='g')
            ax.set_ylim(0, 1)
            #plot horizontal black line at 0.25
            ax.axhline(y=config.CENTERNET.PRED_HEATMAP_THR, color='k', linestyle='--', label='selection threshold')
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

        
    pass