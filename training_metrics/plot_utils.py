from io import BytesIO
import re
import time
from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import os

from matplotlib.transforms import Bbox
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG
from IPython.display import SVG

def plot_sweeps(ax, radar_sw, nr_sweeps=6, show_vels=False):
    assert radar_sw.shape[0] >= nr_sweeps, "radar_sw.shape[0] != nr_sweeps"

    colors = ['#00FF00', '#00D815', '#00B13C', '#008A63', '#00638A', '#0000FF'][::-1]
    # colors = ['#00FF00', '#00D815', '#00B13C', '#008A63', '#00638A', '#003CB1', '#0015D8', '#0000FF'][::-1]
    ang_bins = np.linspace(0, 2 * np.pi, radar_sw.shape[1], endpoint=False)

    for idx in reversed(range(nr_sweeps)):
        #points
        ax.scatter(ang_bins, 
                    radar_sw[idx][:, 0], 
                    marker='o', s=7, c=colors[idx], 
                    alpha=1 - 0.9 * idx / nr_sweeps,
                    label='t sweep' if idx == 0 else f't - {(idx)} sweep')

        if show_vels:
        # velocities
            ax.quiver(ang_bins, 
                    radar_sw[idx][:, 0], 
                    np.zeros_like(ang_bins),
                    radar_sw[idx][:, 1],
                    angles='xy', scale_units='xy', scale=1, #nusc scale=1/6
                    color='k', 
                    alpha=1 - 0.9 * idx / nr_sweeps,
                    label='radial velocity')


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

def plot_bboxes(ax, anns, ground_truth=True, vt=False, tracking=False, ann_ids=None, show_match_radius=None):
    if ground_truth is True:
        color = 'r'
        line_label = 'target'
    else:
        color = 'k'
        line_label = 'prediction'

    # ground truth: ANN_FEATURE_SIZE = 17
    # range - m
    # angle - radians
    # orientation - front relative to ray - radians
    # size - [w l] - relative to range
    # velocity - [vr vt] - relative to range
    # ----------------------------------------------------------------
    # segment - [r_min th_min r_max th_max] ###   preds: hmap/conf
    # visibility - 0-4
    # class - 0-22
    # attribute - 0-8
    # num_radar_pts
    # num_lidar_pts

    #pred merge
    #  if ground_truth is False:
    #     #for every continous zone with adjacent angles:
    #     #take the mean of the range and angle
    #     new_anns = np.zeros((0, anns.shape[1]))

    #     cx, cy = anns[:, 0] * np.cos(anns[:, 1]), anns[:, 0] * np.sin(anns[:, 1])
    #     while anns.shape[0] > 0:
    #         mask = np.abs(cx - cx[0], cy - cy[0]) < 2
    #         new_anns = np.vstack((new_anns, np.mean(anns[mask], axis=0)))
    #         anns = anns[~mask]
    #         cx, cy = cx[~mask], cy[~mask]
    
    #     anns = new_anns
    
    anns = np.array(anns, dtype=np.float32)

    car_ori = anns[:, 2] + anns[:, 1]
    car_size = anns[:, 3:5] * anns[:, 0].reshape(-1, 1)
    car_range = anns[:, 0]
    car_rot = anns[:, 1]

    if ground_truth is False:
        # pred_alpha = np.ones((anns.shape[0]))
        pred_alpha = anns[:, -1]
        # ax.scatter(car_rot, car_range, marker='x', color='k', s=15)
    else:
        pred_alpha = np.ones((anns.shape[0]))
        if show_match_radius is not None:
            #plot circles of radius 4 around each radar point
            for x in range(anns.shape[0]):
                circle = plt.Circle(
                    (car_range[x] * np.cos(car_rot[x]), 
                    car_range[x] * np.sin(car_rot[x])),
                    show_match_radius, color='k', alpha=0.1, transform=ax.transData._b)
                ax.add_artist(circle)

    car_corners = get_car_corners(car_size, car_range, car_ori, car_rot)

    for x in range(car_corners.shape[0]):
        gigi = car_corners[x]
        gigi = np.vstack((gigi, gigi[0]))
        colors = ['g', 'b', 'y', 'c', 'k']
        if ground_truth is True:
            has_radar_point = anns[x, 14] >= 1
            ax.plot(gigi[:, 1],
                    gigi[:, 0], 
                    c=color, 
                    label='target' if has_radar_point else 'no radar reading target',
                    # label=line_label,
                    alpha=(1 if has_radar_point else 0.3)) #colored if has a radar point
        else:
            if ann_ids is not None:
                ax.plot(gigi[:, 1], gigi[:, 0], c=colors[hash(ann_ids[x]) % 5],
                        label=line_label, alpha=1)
                # ax.plot(gigi[:, 1], gigi[:, 0], c=colors[hash(ann_ids[x]) % 5],
                #         label=line_label, alpha=pred_alpha[x])
            else:
                ax.plot(gigi[:, 1], gigi[:, 0], c=color,
                        label=line_label, alpha=pred_alpha[x])


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


def do_plots(loaded_data, scene_name, sweeps, show_vels=False, ground_truth=True, predictions=False, vt=False, tracking=False, ego_track=None, show_match_radius=None, add_legend=False, make_video=False):
    print("scene_name: ", scene_name)
    start_time = time.time()

    assert scene_name in loaded_data, "scene_name not in loaded_data"
    if make_video:
        os.makedirs(f'./plots/{scene_name}', exist_ok=True)
    
    scene = loaded_data[scene_name]
    
    if tracking is True:
        assert ego_track is not None, "ego_track is None"
        track_ids, _ = gen_tracks(scene['preds'], ego_track)

    for sw in sweeps:
        if sw < 0 or sw >= len(scene['radar']):
            continue
        radar_sw = scene['radar'][sw]
        # anns = scene['anns'][sw][:, :-1]  #TODO: why?
        anns = scene['anns'][sw]
    
        if predictions is True:
            preds = scene['preds'][sw]
        
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 12))
        ax.set_ylim(0, 50)
        ax.set_title(f"scene: {scene_name}, sweep: {sw}, time: {scene['timestamp'][sw]}")
        
        # ax.set_title(f"scene: {scene_name}, sweep: {sw}")
        plot_sweeps(ax, radar_sw, show_vels=show_vels)
        if ground_truth:
            # plot_bboxes(ax, anns, ground_truth=ground_truth, tracking=tracking, ann_ids=scene['anns_tokens'][sw])
            plot_bboxes(ax, anns, ground_truth=ground_truth, tracking=tracking, vt=vt, show_match_radius=show_match_radius)
            pass
        if predictions:
            plot_bboxes(ax, preds, ground_truth=False, tracking=tracking, vt=vt, ann_ids=track_ids[sw] if tracking is True else None, show_match_radius=show_match_radius)
                        
        if ego_track is not None:
            trans, rot = ego_track
            trans = trans - trans[sw]
            
            trans = (rot[sw].T @ trans.T).T
            #transform trans to polar coordinates
            # print(trans)
            # print(np.atan2(trans[:, 1], trans[:, 0]), np.hypot(trans[:, 1], trans[:, 0]))
            ax.plot(np.atan2(trans[:, 1], trans[:, 0]), np.hypot(trans[:, 0], trans[:, 1]), marker='>', c='k', label='ego track', lw=0.2, markersize=2)
        
        #reduce double labels
        if add_legend:
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            #sort labels alphabetically:
            labels = np.array(list(by_label.values()))
            handles = np.array(list(by_label.keys()))
            ord_labels = np.argsort([re.sub(r'[^a-zA-Z]', '', x) for x in handles])
            # plt.legend(labels[ord_labels], handles[ord_labels], loc='center left', fontsize='medium', ncols = 2)
            plt.legend(labels[ord_labels], handles[ord_labels], loc='upper right', bbox_to_anchor=(1, 1), fontsize='medium', ncols = 2)


        if make_video:
            plt.savefig(f'./plots/{scene_name}/' + str(sw) + '.png', bbox_inches='tight')
            plt.savefig(f'./plots/{scene_name}/' + str(sw) + '.svg', bbox_inches='tight')
        else:
            plt.show()
        plt.close()

        #GT HEATMAP
        # plt.figure(figsize=(12, 3))
        # plt.plot(np.arange(360), scene['heatmap'][sw], label='ground truth heatmap', c='r')
        # plt.xlabel('angle')
        # plt.show()
    
    if make_video:
        print(f"Time taken: {time.time() - start_time}")


#for a single scene
def anns_to_cart(anns):
    #range, angle, orientation, w, l, vr, vt, hmap
    car_ori = anns[:, 2] + anns[:, 1]
    car_size = anns[:, 3:5] * anns[:, 0].reshape(-1, 1)
    car_range = anns[:, 0]
    car_rot = anns[:, 1]

    car_trans = np.array([np.array(
        [car_range[x] * np.cos(car_rot[x]), 
         car_range[x] * np.sin(car_rot[x]), 
         0]) for x in range(car_range.shape[0])])

    # car_trans = np.array(
    #     [[car_range[x] * np.cos(car_rot[x]), 
    #      car_range[x] * np.sin(car_rot[x]), 
    #      0] for x in range(car_range.shape[0])])

    car_ori_matrix = np.array([
        [[np.cos(car_ori[x]), -np.sin(car_ori[x]), 0],
         [np.sin(car_ori[x]), np.cos(car_ori[x]),  0],
         [0,                  0,                   1]] for x in range(car_ori.shape[0])])

    car_rot_matrix = np.array([
        [[np.cos(car_rot[x]), -np.sin(car_rot[x]), 0],
         [np.sin(car_rot[x]), np.cos(car_rot[x]),  0],
         [0,                  0,                   1]] for x in range(car_rot.shape[0])])
    
    #car velocity in cartesian coordinates
    car_vr = anns[:, 5] * car_range
    car_vt = anns[:, 6] * car_range
    
    car_v_endp = np.stack([car_vr + car_range, car_vt], axis=1)
    car_v_endp = np.array([(car_rot_matrix[x][:2, :2] @ car_v_endp[x].T).T for x in range(car_v_endp.shape[0])])
    
    car_v_endp -= car_trans[:, :2]

    # car_v_endp[:, 2] = 0

    return car_trans, car_ori_matrix, car_size, car_v_endp

# receives all anns from a whole scene
def gen_tracks(all_anns, ego_track):
    #anns: [[range, angle, orientation, w, l, vr, vt, hmap], ...]

    # plt.figure(figsize=(12, 12))
    # plt.title('Initial')
    # plt.xlim(-50, 50)
    # plt.ylim(-50, 50)
    # gigi = np.linspace(0, 2 * np.pi, 100)
    # for x in range(6):
    #     plt.plot(10 * x * np.cos(gigi), 10 * x *  np.sin(gigi), 'k', alpha=0.3)

    global_anns = [(np.empty((0, 3)), np.empty((0, 3, 3)), np.empty((0, 2)), np.empty((0, 2)))]
    trans, rot = ego_track

    for idx, anns in enumerate(all_anns):
        if idx == 0:
            continue
        
        if len(anns) == 0:
            global_anns.append(global_anns[0])
            continue

        cart_trans, cart_ori, cart_wl, cart_v = anns_to_cart(anns)

        assert len(cart_trans) == len(cart_ori) == len(cart_wl) == len(cart_v), "not all scene anns lens equal"

        cart_ori = np.array([rot[idx] @ x for x in cart_ori])
        cart_v = np.array([rot[idx][:2, :2] @ x for x in cart_v])
        cart_trans = (rot[idx] @ cart_trans.T).T
        cart_trans += trans[idx]

        #Center
        # plt.scatter(cart_trans[:, 0], cart_trans[:, 1], c='g', label='pred', marker='x')
        
        #Velocity
        # plt.quiver(cart_trans[:, 0], cart_trans[:, 1], cart_v[:, 0], cart_v[:, 1], angles='xy', scale_units='xy', scale=2, color='k')#, width = 0.005, headwidth=0, headaxislength=0)

        #Orientation
        # plt.quiver(cart_trans[:, 0], cart_trans[:, 1], cart_ori[:, 0, 0], cart_ori[:, 1, 0], angles='xy', scale_units='xy', scale=1/5, color='r') 

        global_anns.append((cart_trans, cart_ori, cart_wl, cart_v))
    
    ids = []
    to_give = 0
    for idx, anns in enumerate(all_anns):
        if idx == 0:
            ids.append([])
            continue
        desc_conf = np.argsort(anns[:, -1])[::-1]
        last_anns = global_anns[idx - 1]
        cur_anns = global_anns[idx]
        last_anns_mask = np.ones(len(last_anns[0]), dtype=bool)
        new_ids = np.array([-1] * len(cur_anns[0]))
        for x in desc_conf:
            diffs = last_anns[0] - cur_anns[0][x]
            diffs = diffs[:, :2] + 0.5 * last_anns[3] #TODO: get real time diff

            diffs = np.hypot(diffs[:, 0], diffs[:, 1])
            ord_best = np.argsort(diffs)
            found = 0
            for best in ord_best:
                if diffs[best] <= max(cur_anns[2][x][0], cur_anns[2][x][1]) and last_anns_mask[best]: #TODO max or sqrt like centertrack?
                    new_ids[x] = ids[idx - 1][best]
                    last_anns_mask[best] = False
                    found = 1
                    break
            if found == 0:
                new_ids[x] = to_give
                to_give += 1
        ids.append(new_ids)
        assert len(ids[-1]) == len(cur_anns[0]), "len(ids[-1]) != len(cur_anns[0])"
    return ids, global_anns
