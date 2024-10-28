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


def extract_pred_anns(output):
    list_anns = np.empty((0, 4))
    for i in range(output.size(0)):
        gigi = output[i, :, :].detach().cpu().numpy()
        # anns_i = []
        for j in range(360):
            if  gigi[j, 0] >= 0.25 and  gigi[j, 0] >  gigi[(j + 359) % 360, 0] and  gigi[j, 0] >  gigi[(j + 1) % 360, 0]:
                r = gigi[j, 1]
                th = j
                # anns_i.append((r * np.cos(th * np.pi / 180), r * np.sin(th * np.pi / 180), i, gigi[j, 0]))
                list_anns = np.vstack((list_anns, np.array([r * np.cos(th * np.pi / 180), r * np.sin(th * np.pi / 180), i, gigi[j, 0]])))

        # list_anns.append(anns_i)
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
            # print(gt_idx)
            # print(gt_box)

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

            # Since it is a match, update match data also.
            # gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]

            # match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            # match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            # match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            # period = np.pi if class_name == 'barrier' else 2 * np.pi
            # match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))

            # match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            # match_data['conf'].append(pred_box.detection_score)

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
        
        conf.append(pred_box[3])

    return tp, fp, conf

    # Check if we have any matches. If not, just return a "no predictions" array.
    # if len(match_data['trans_err']) == 0:
    #     return DetectionMetricData.no_predictions()

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    # Accumulate.
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)

    # print('prec:', prec)
    # print('rec:', rec)

    rec_interp = np.linspace(0, 1, 101)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    # import matplotlib.pyplot as plt
    # plt.plot(rec[10:], prec[10:])
    # plt.plot(rec, prec)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title(f'Precision-Recall Curve for threshold {dist_th}')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.grid()
    # plt.show()
    # plt.plot(rec, conf)
    # plt.show()

    # import sys
    # sys.exit()

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    # for key in match_data.keys():
    #     if key == "conf":
    #         continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

    #     else:
    #         # For each match_data, we first calculate the accumulated mean.
    #         tmp = cummean(np.array(match_data[key]))

    #         # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
    #         match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    # return DetectionMetricData(recall=rec,
    #                            precision=prec,
    #                            confidence=conf,
    #                            trans_err=match_data['trans_err'],
    #                            vel_err=match_data['vel_err'],
    #                            scale_err=match_data['scale_err'],
    #                            orient_err=match_data['orient_err'],
    #                            attr_err=match_data['attr_err'])
    return prec, rec, conf


# def calc_ap2(precision, min_recall: float, min_precision: float) -> float:
#     # Calculated average precision. 

#     assert 0 <= min_precision < 1
#     assert 0 <= min_recall <= 1

#     prec = np.copy(precision)
#     prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
#     prec -= min_precision  # Clip low precision
#     prec[prec < 0] = 0
#     return float(np.mean(prec)) / (1.0 - min_precision)


# def calc_sweep_metrics(output, target):
#     gt_anns = extract_gt_anns(target) # [[(x, y, sample_num)], ...]
#     pred_anns = extract_pred_anns(output) # [(x, y, sample_num, confidence), ...]

#     # print(gt_anns)
#     # print(pred_anns)
#     # print(pred_anns.shape)
#     # print(len(gt_anns))
#     # print(len(pred_anns))

#     dist_thresholds = [0.5, 1.0, 2.0, 4.0]
#     aps = []

#     for dist_thresh in dist_thresholds:
#         prec, rec, conf = accumulate(deepcopy(gt_anns.copy), pred_anns.copy(), dist_thresh, verbose=False)
#         ap = calc_ap2(prec, 0.1, 0.1)
#         aps.append(ap)

#     # print(aps)
#     # print(np.mean(aps))
#     return np.mean(aps)


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


def get_tp_fp_conf(output, target, dist_thresholds):
    gt_anns = extract_gt_anns(target) # [[(x, y, sample_num)], ...]
    pred_anns = extract_pred_anns(output) # [(x, y, sample_num, confidence), ...]


    npos = np.sum([len(x) for x in gt_anns])

    tp_fp_conf = []

    for dist_thresh in dist_thresholds:
        tp, fp, conf = accumulate(deepcopy(gt_anns), pred_anns.copy(), dist_thresh, verbose=False)
        tp_fp_conf.append((tp, fp, conf))

    tp_fp_conf = np.array(tp_fp_conf)
    # print(npos)
    # print(len(pred_anns))
    # print(tp_fp_conf.shape)
    
    return tp_fp_conf, npos