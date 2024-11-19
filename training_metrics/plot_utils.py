import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import os

def plot_sweeps(ax, radar_sw, nr_sweeps=6, show_vels=False):
    assert radar_sw.shape[0] >= nr_sweeps, "radar_sw.shape[0] != nr_sweeps"

    colors = ['#00FF00', '#00D815', '#00B13C', '#008A63', '#00638A', '#003CB1', '#0015D8', '#0000FF'][::-1]
    ang_bins = np.linspace(0, 2 * np.pi, radar_sw.shape[1], endpoint=False)

    for idx in reversed(range(nr_sweeps)):
        #points
        ax.scatter(ang_bins, 
                    radar_sw[idx][:, 0], 
                    marker='o', s=7, c=colors[idx], 
                    alpha=1 - 0.9 * idx / nr_sweeps)

        if show_vels:
        # velocities
            ax.quiver(ang_bins, 
                    radar_sw[idx][:, 0], 
                    np.zeros_like(ang_bins),
                    radar_sw[idx][:, 1],
                    angles='xy', scale_units='xy', scale=1/6,
                    color='k', 
                    alpha=1 - 0.9 * idx / nr_sweeps)


def plot_bboxes(ax, anns, ground_truth=True, vt=False, tracking=False, ann_ids=None):
    if ground_truth is True:
        color = 'r'
        line_label = 'target'
    else:
        color = 'g'
        line_label = 'prediction'

    # ground truth: ANN_FEATURE_SIZE = 17
    # range - m
    # angle - radians
    # orientation - front relative to ray - radians
    # size - [w l] - relative to range
    # velocity - [vr vt] - relative to range
    # segment - [r_min th_min r_max th_max]
    # visibility - 0-4
    # class - 0-22
    # attribute - 0-8
    # num_radar_pts
    # num_lidar_pts

    anns = np.array(anns)

    car_ori = anns[:, 2] + anns[:, 1]
    # car_ori = np.arcsin(anns[:, 2]) + anns[:, 1]
    car_size = anns[:, 3:5] * anns[:, 0].reshape(-1, 1)
    car_range = anns[:, 0]

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
    car_trans = np.array([np.array([car_range[x] * np.cos(anns[x, 1]), car_range[x] * np.sin(anns[x, 1])]) for x in range(car_corners.shape[0])])
    

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
        colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w']
        if ann_ids is not None:
            ax.plot(gigi[:, 1], gigi[:, 0], c=colors[hash(ann_ids[x]) % 8], label=line_label)
        else:
            ax.plot(gigi[:, 1], gigi[:, 0], c=color, label=line_label)


    for x in range(car_range.shape[0]):
        if ground_truth is False:
            color = 'k'

        ax.arrow(anns[x, 1], car_range[x], 0, car_range[x] * anns[x, 5], lw=4, fc=color, ec=color, head_width=0, head_length=0, alpha=0.5)
        
        if ground_truth is True or tracking is True or vt is True:
            ranges = [car_range[x], np.hypot(car_range[x], anns[x, 6] * car_range[x])]
            angles = [anns[x, 1], anns[x, 1] + np.arcsin(anns[x, 6] * car_range[x] / ranges[1])]       
            ax.arrow(angles[0], ranges[0], angles[1] - angles[0], ranges[1] - ranges[0], lw=4, fc=color, ec=color, head_width=0, head_length=0, alpha=0.5)


def get_ego_track_cart(loaded_data, scene_name):
    scene = loaded_data[scene_name]
    trans = np.array(scene['ego_translation'])
    trans = (trans - trans[0])
    trans[:, 2] = 0

    # print(trans)
    inv_rot0 = Quaternion(scene['ego_rotation'][0]).rotation_matrix.T
    # print("inv_rot0: ", inv_rot0)

    #multiply each translation by the rotation matrix inv_rot0
    trans = (inv_rot0 @ trans.T).T
    rot = np.array([inv_rot0 @ Quaternion(x).rotation_matrix for x in scene['ego_rotation']])

    # plt.figure(figsize=(8, 8))
    # plt.grid()
    # plt.xlabel('x(m)')
    # plt.ylabel('y(m)')
    # plt.title("Ego Translation")
    # maxlim = max(10, np.max(np.abs(trans) + 5))
    # plt.ylim(-maxlim, maxlim)
    # plt.xlim(-maxlim, maxlim)
    # plt.scatter(trans[:, 0], trans[:, 1], marker='x', c='r')
    # plt.show()
    
    return trans, rot


def do_plots(loaded_data, scene_name, sweeps, show_vels=False, ground_truth=True, predictions=False, vt=False, tracking=False, ego_track=None, make_video=False):
    print("scene_name: ", scene_name)

    assert scene_name in loaded_data, "scene_name not in loaded_data"
    if make_video:
        os.makedirs(f'./plots/{scene_name}', exist_ok=True)
    
    scene = loaded_data[scene_name]
    
    for sw in sweeps:
        if sw < 0 or sw >= len(scene['radar']):
            continue
        radar_sw = scene['radar'][sw]
        anns = scene['anns'][sw][:, :-1]
    
        if predictions is True:
            preds = scene['preds'][sw]
        
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 12))
        ax.set_ylim(0, 50)
        ax.set_title(f"scene: {scene_name}, sweep: {sw}, time: {scene['timestamp'][sw]}")
        # ax.set_title(f"scene: {scene_name}, sweep: {sw}")
        plot_sweeps(ax, radar_sw, show_vels=show_vels)
        if ground_truth:
            # plot_bboxes(ax, anns, ground_truth=ground_truth, tracking=tracking, ann_ids=scene['anns_tokens'][sw])
            plot_bboxes(ax, anns, ground_truth=ground_truth, tracking=tracking, vt=vt)
        if predictions:
            plot_bboxes(ax, preds, ground_truth=False, tracking=tracking, vt=vt)
                        
        if ego_track is not None:
            trans, rot = ego_track
            trans = trans - trans[sw]
            
            trans = (rot[sw].T @ trans.T).T
            #transform trans to polar coordinates
            # print(trans)
            # print(np.atan2(trans[:, 1], trans[:, 0]), np.hypot(trans[:, 1], trans[:, 0]))
            ax.plot(np.atan2(trans[:, 1], trans[:, 0]), np.hypot(trans[:, 0], trans[:, 1]), marker='>', c='k', label='ego track', lw=0.2, markersize=2)

        if make_video:
            plt.savefig(f'./plots/{scene_name}/' + str(sw) + '.png')
            plt.close()
