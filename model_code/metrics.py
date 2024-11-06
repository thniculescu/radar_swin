from typing import List, Callable
import numpy as np
from copy import deepcopy

def extract_gt_anns(target):
    #for each position where target[:, 8] == 1, extract the corresponding target[:, 1:8]
    list_anns = []
    for i in range(target.size(0)):
        anns_i = []
        gigi = target[i, :, :].detach().cpu().numpy()
        for j in range(360):
            if gigi[j, 8] == 1:
                r = gigi[j, 1]
                th = j
                #anns append: x, y cartesian from polar r th, sample_num = i:
                anns_i.append((r * np.cos(th * np.pi / 180), r * np.sin(th * np.pi / 180), i))
        list_anns.append(anns_i)
    return list_anns


def extract_pred_anns(output, pred_thresh):
    list_anns = np.empty((0, 4))
    for i in range(output.size(0)):
        gigi = output[i, :, :].detach().cpu().numpy()

        mask = gigi[:, 0] >= pred_thresh
        prev_values = np.roll(gigi[:, 0], 1)
        next_values = np.roll(gigi[:, 0], -1)

        local_max = mask & (gigi[:, 0] > prev_values) & (gigi[:, 0] > next_values)

        indices = np.where(local_max)[0]

        r_values = gigi[indices, 1]
        thetas = indices * np.pi / 180
        x_coords = r_values * np.cos(thetas)
        y_coords = r_values * np.sin(thetas)

        new_anns = np.column_stack((x_coords, y_coords, np.full_like(indices, i), gigi[indices, 0]))

        list_anns = np.vstack((list_anns, new_anns))

    return list_anns


def accumulate(gt_boxes: List,
               pred_boxes: List,
               dist_th: float,
               verbose: bool = False):

    # Average Precision over predefined different recall thresholds for a single distance threshold.
    # The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    # :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    # :param pred_boxes: Maps every sample_token to a list of its sample_results.
    # :param class_name: Class to compute AP on.
    # :param dist_fcn: Distance function used to match detections and ground truths.
    # :param dist_th: Distance threshold for a match.
    # :param verbose: If true, print debug messages.
    # :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.

    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    # Count the positives.

    npos = np.sum([len(x) for x in gt_boxes])
    # print(len(gt_boxes))
    # print(npos)

    # For missing classes in the GT, return a data structure corresponding to no predictions.

    # Organize the predictions in a single list.
    pred_boxes_list = pred_boxes

    # Sort by confidence.
    # sort pred_boxes_list by confidence = column 3 decreasing
    pred_boxes_list = pred_boxes_list[pred_boxes_list[:, 3].argsort()[::-1]] 
    
    # print(pred_boxes_list)

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': []}

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    # taken = set()  # Initially no gt bounding box is matched.
    for x in range(pred_boxes_list.shape[0]):
        pred_box = pred_boxes_list[x]
        min_dist = 200  # Initialize to a large number.
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[int(pred_box[2])]):

            # Find closest match among ground truth boxes
            this_distance = np.linalg.norm(np.array([gt_box[0], gt_box[1]]) - np.array([pred_box[0], pred_box[1]]))

            if this_distance < min_dist:
                min_dist = this_distance
                match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        if is_match:
            # taken.add((pred_box.sample_token, match_gt_idx))
            #remove the matched ground truth box from the list of ground truth boxes
            gt_boxes[int(pred_box[2])].pop(match_gt_idx)

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
        
        conf.append(pred_box[3])

    return tp, fp, conf

def calc_ap(thresh_tp_fp_conf, num_gt, ap_dist_thresh, min_precision=0.1, min_recall=0.1):
    aps = []
    print(thresh_tp_fp_conf.shape)

    assert len(ap_dist_thresh) == len(thresh_tp_fp_conf)
    
    for i in range(len(thresh_tp_fp_conf)):
        tp_fp_conf = np.array(thresh_tp_fp_conf[i, :, :])
        # print(tp_fp_conf)

        #sort by confidence decreasing
        tp_fp_conf = tp_fp_conf[:, tp_fp_conf[2, :].argsort()[::-1]]
        # print(tp_fp_conf)

        tp = np.cumsum(tp_fp_conf[0, :]).astype(float)
        fp = np.cumsum(tp_fp_conf[1, :]).astype(float)
        prec = tp / (tp + fp)
        rec = tp / float(num_gt)

        rec_interp = np.linspace(0, 1, 101)  # 101 steps, from 0% to 100% recall.
        prec = np.interp(rec_interp, rec, prec, right=0)
        conf = np.interp(rec_interp, rec, tp_fp_conf[2, :], right=0)
        rec = rec_interp

        assert 0 <= min_precision < 1
        assert 0 <= min_recall <= 1

        prec_i = np.copy(prec)
        prec_i = prec_i[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.

        prec_i -= min_precision  # Clip low precision
        prec_i[prec_i < 0] = 0
        ap = float(np.mean(prec_i)) / (1.0 - min_precision)
        aps.append(ap)

        # import matplotlib.pyplot as plt
        # plt.plot(rec, prec, label=f'threshold {ap_dist_thresh[i]}')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Precision-Recall Curves')
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        # #plot horizontal black line at 0.1
        # plt.axhline(y=0.1, color='k', linestyle='--')
        # plt.axvline(x=0.1, color='k', linestyle='--')
        # plt.legend()


    return np.array(aps)


def get_tp_fp_conf(output, target, dist_thresholds, pred_thresh):
    gt_anns = extract_gt_anns(target) # [[(x, y, sample_num)], ...]
    pred_anns = extract_pred_anns(output, pred_thresh) # [(x, y, sample_num, confidence), ...]


    npos = np.sum([len(x) for x in gt_anns])

    tp_fp_conf = np.empty((len(dist_thresholds), 3, len(pred_anns)))

    for i, dist_thresh in enumerate(dist_thresholds):
        tp, fp, conf = accumulate(deepcopy(gt_anns), pred_anns, dist_thresh, verbose=False)
        # tp_fp_conf.append((tp, fp, conf))
        tp_fp_conf[i, 0, :] = tp
        tp_fp_conf[i, 1, :] = fp
        tp_fp_conf[i, 2, :] = conf

    # tp_fp_conf = np.array(tp_fp_conf)
    
    return tp_fp_conf, npos