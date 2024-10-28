# %%
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix, view_points
import os, json, time, timeit
from multiprocessing import Pool
from functools import partial, reduce
from utils import plot_proc_data, calc_ego_vel, split_time, split_ang_bins, filter_boxes, get_points, CATEGORY_NAMES, ATTRIBUTE_NAMES, ANN_FEATURE_SIZE, COMP_TYPE
# %matplotlib inline

# %%
os.chdir('./viz_preproc')

with open('../run_paths.json') as f:
    paths = json.load(f)

# DATASET_PATH = paths['nusc-trainval']
DATASET_PATH = paths['tmp']
DATASET_NAME = 'v1.0-trainval'

# DATASET_PATH = paths['nusc-mini']
# DATASET_NAME = 'v1.0-mini'

OUT_PREPROC_PATH = './output_preproc'

#Keep only moving targets
RadarPointCloud.default_filters()
# RadarPointCloud.dynprop_states = [0, 2, 6]
# RadarPointCloud.disable_filters()

# %%
nusc = NuScenes(version=DATASET_NAME, dataroot=DATASET_PATH, verbose=True)


# DEBUG check how many points are used
# total_radar_points = 0
# total_used_radar_points = 0
# %%
def preproc_scene(scene_id, nr_samps = 50, plots=False, max_range=50, nr_sweeps=6, ang_bins=360, min_radar_pts=1, min_lidar_pts=1, save_output=True, out_path=None, vis_level=0, only_vehicle=True, only_moving=False):
    if save_output:
        if out_path is None:
            out_path = './output_preproc'
        os.makedirs(out_path, exist_ok=True)

    start_time = timeit.default_timer()

    scene_name = nusc.scene[scene_id]['name'].split('-')[1]
    print(scene_name)
    # result = {scene_name: { 'radar' : np.empty((0, 6, ang_bins, 3)), 'anns' : np.empty((0, ANN_FEATURE_SIZE)), 'ego_vels' : np.empty((0, 2)), 'heatmap' : np.empty((0, ang_bins))}}
    result = {scene_name: { 'radar' : [], 'anns' : [], 'ego_vels' : [], 'heatmap' : []}}
    
    cur_sample = nusc.get('sample', nusc.scene[scene_id]['first_sample_token'])
    global_ego_vel = calc_ego_vel(nusc, cur_sample)

    cur_sample = nusc.get('sample', cur_sample['next'])
    t0 = cur_sample['timestamp']
    
    for nr_samps in range(min(nr_samps, nusc.scene[scene_id]['nbr_samples'] - 2)):
        # if nr_samps not in range(7, 8):
        #     cur_sample = nusc.get('sample', cur_sample['next'])
        #     continue

        # print('Sample:', nr_samps + 1)
        
        ref_chan = 'LIDAR_TOP'
        ref_sd_record = nusc.get('sample_data', cur_sample['data'][ref_chan])

        # calculate ego velocity
        cur_yaw = Quaternion(nusc.get('ego_pose', ref_sd_record['ego_pose_token'])['rotation']).yaw_pitch_roll[0]
        cur_velocity = global_ego_vel[1][np.argmin(np.abs(global_ego_vel[0] - cur_sample['timestamp'] / 1e6))]
        cur_velocity = np.dot(Quaternion(scalar=np.cos(cur_yaw / 2), vector=[0, 0, np.sin(cur_yaw / 2)]).inverse.rotation_matrix, cur_velocity)
        cur_velocity = cur_velocity[:2]

        sweep_points = [np.empty((0, 5)) for i in range(nr_sweeps)] # for each "sweep", concatenate the points from diff channels, t -> t - nr_sweeps + 1
        sweep_frame = np.empty((nr_sweeps, ang_bins, 3)) #range, speed, rcs

        # if plots:
        #     fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        #     ax.axis('on')
        #     ax.grid()
        #     ax.set_xticks(np.arange(-max_range, max_range, 10))
        #     ax.set_yticks(np.arange(-max_range, max_range, 10))
        #     ax.set_xlim(-max_range, max_range)
        #     ax.set_ylim(-max_range, max_range)
        #     gigel = np.arange(0, 360, 2)
        #     gigel = np.radians(gigel)
        #     for gogu in range(1, max_range // 10 + 1):
        #         ax.plot(np.cos(gigel) * gogu * 10, np.sin(gigel) * gogu * 10, 'k--', linewidth=0.5)


        for chan_type, chan_tok in cur_sample['data'].items():
            if 'RADAR' in chan_type:

                # if plots:
                #     nusc.render_sample_data(chan_tok, ax=ax, nsweeps=nr_sweeps, axes_limit=max_range, underlay_map=True, with_anns=True, verbose=False)
                #     ax.set_title('Sample: ' + str(nr_samps + 1) + ', nsweeps: ' + str(nr_sweeps) + ', Comp: ' + COMP_TYPE + ', Time: ' + str((cur_sample['timestamp'] - t0) / 1e6))



                points, times, test_pc = get_points(nusc, cur_sample, chan_type, chan_tok, ref_chan, ref_sd_record, nsweeps=nr_sweeps, speed_ref=COMP_TYPE, max_range=max_range, ego_vels=global_ego_vel)
                
                # Check speed angles with respect to ego vehicle ./test_viteze
                # check_vit_poz_test = np.append(check_vit_poz_test, test_pc[:, [0, 1, 6, 7]], axis=0)
                
                # gigel = np.arange(0, 360, 2)
                # gigel = np.radians(gigel)
                # ax.plot(0, 0, 'kx')
                # for gogu in range(1, 7):
                #     ax.plot(np.cos(gigel) * gogu * 10, np.sin(gigel) * gogu * 10, 'k--', linewidth=0.5)
                # ax.plot(test_pc[:, 0], test_pc[:, 1], 'b.', markersize=3)
                
                # for i in range(test_pc.shape[0]):
                #     ax.arrow(test_pc[i, 0], test_pc[i, 1], test_pc[i, 6], test_pc[i, 7], head_width=0.5, head_length=0.5, fc='r', ec='r')
                #     # ax.arrow(test_pc2[i, 0], test_pc2[i, 1], test_pc2[i, 6], test_pc2[i, 7], head_width=0.5, head_length=0.5, fc='r', ec='g')

                points, times = split_time(points, times)
                for idx,p in enumerate(points):
                    sweep_points[idx] = np.append(sweep_points[idx], p, axis=0)



                
        # if plots:
            # if not os.path.exists('plots/' + nusc.scene[scene_id]['name']):
            #     os.makedirs('plots/' + nusc.scene[scene_id]['name'])
            # fig.savefig('plots/' + nusc.scene[scene_id]['name'] + '/' + str(nr_samps) + '_nrsweeps' + str(nr_sweeps) + '.png')
            # plt.close(fig)
        

        for sw, p in enumerate(sweep_points):
            binned_points, used_points = split_ang_bins(p, nr_ang_bins=ang_bins)
            sweep_frame[sw] = binned_points

            #DEBUG check how many points are used
            # total_radar_points += p.shape[0]
            # total_used_radar_points += used_points
        

        # FILTER ANNOTATIONS
        filtered_anns = filter_boxes(nusc, cur_sample['anns'],
                                     vis_level=vis_level,
                                     only_vehicle=only_vehicle,
                                     only_moving=only_moving,
                                     min_radar_pts=min_radar_pts,
                                     min_lidar_pts=min_lidar_pts,
                                     skip_filter=False)

        # go through sample, get all annotation positions and box sizes
        _, boxes, _ = nusc.get_sample_data(cur_sample['data'][ref_chan], selected_anntokens=filtered_anns, use_flat_vehicle_coordinates=True)
        sweep_anns = np.empty((0, ANN_FEATURE_SIZE))

        box_corners = np.empty((0, 4, 2))
        box_segments = np.empty((0, 6))

        for box, ann in zip(boxes, filtered_anns):

            box.velocity = nusc.box_velocity(ann)
            box.velocity = np.nan_to_num(box.velocity)

            r = np.hypot(box.center[0], box.center[1])

            if r > max_range:
                continue

            th_rad = np.arctan2(box.center[1], box.center[0])
            #create quaternion from angle
            rot_quat = Quaternion(axis=[0, 0, 1], radians=th_rad)

            rel_orient_quat = (rot_quat.inverse * box.orientation)
            if rel_orient_quat.axis[2] == -1:
                rel_orient_rad = -(rel_orient_quat.radians)
            else:
                rel_orient_rad = rel_orient_quat.radians

            rel_size = box.wlh[:2] / r
            #box velovity in ego coord, not taking into account ego movement
            box.velocity = np.dot(Quaternion(scalar=np.cos(cur_yaw / 2), vector=[0, 0, np.sin(cur_yaw / 2)]).inverse.rotation_matrix, box.velocity)
            vel = box.velocity[:2]
            # vel = vel - cur_velocity

            #split velocity into radial and tangential
            vr = np.dot(vel, [np.cos(th_rad), np.sin(th_rad)])
            vt = np.dot(vel, [-np.sin(th_rad), np.cos(th_rad)])
            vel = np.array([vr, vt]) / r # for real sweeps

            # ground truth: ANN_FEATURE_SIZE = 16
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


            # maybe?? bin angle offset

            relevant_corners = box.bottom_corners().T[:, :2]

            box_corners = np.vstack((box_corners, relevant_corners[None, :, :]))

            relevant_rs = np.hypot(relevant_corners[:, 0], relevant_corners[:, 1])
            relevant_ths = np.arctan2(relevant_corners[:, 1], relevant_corners[:, 0])
            # correction for discontinuity PI/-PI
            if np.any(relevant_corners[:, 0] < 0):
                if np.any(relevant_corners[:, 1] <= 0) and np.any(relevant_corners[:, 1] > 0):
                    relevant_ths[relevant_ths < 0] += 2 * np.pi

            #keep minimum and maximum relevant th
            min_th = np.argmin(relevant_ths)
            max_th = np.argmax(relevant_ths)
            relevant_ths = np.array([relevant_ths[min_th], relevant_ths[max_th]])
            relevant_rs = np.array([relevant_rs[min_th], relevant_rs[max_th]])

            # OLD HEATMAP
            box_segments = np.vstack((box_segments, [r, th_rad, relevant_rs[0], relevant_ths[0], relevant_rs[1], relevant_ths[1]]))
            
            sample_ann = nusc.get('sample_annotation', ann)
            box_vis = int(sample_ann['visibility_token'])
            box_class = CATEGORY_NAMES[sample_ann['category_name']]

            if len(sample_ann['attribute_tokens']) > 0:
                # print('Attribute:', nusc.get('attribute', sample_ann['attribute_tokens'][0])['name'])
                # print('Attribute:', sample_ann['attribute_tokens'])
                ann_att = nusc.get('attribute', sample_ann['attribute_tokens'][0])['name']
            else:
                ann_att = 'other_obj.static'

            box_att = ATTRIBUTE_NAMES[ann_att]

            sweep_anns = np.vstack((sweep_anns,
                np.hstack((
                    r, th_rad, rel_orient_rad, *rel_size, *vel,
                    relevant_rs[0], relevant_ths[0], relevant_rs[1], relevant_ths[1],
                    box_vis, box_class, box_att, sample_ann['num_radar_pts'], sample_ann['num_lidar_pts']))))

            #TODO: segment discontinuity
            # if relevant_ths[0] < np.pi and relevant_ths[1] > np.pi:
            #     new_r = relevant_rs[0] + (relevant_rs[1] - relevant_rs[0]) * (np.pi - relevant_ths[0]) / (relevant_ths[1] - relevant_ths[0])
            #     box_segments = np.vstack((box_segments, [r, th_rad, relevant_rs[0], relevant_ths[0], relevant_rs[1], relevant_ths[1]]))
            # else:
            #     box_segments = np.vstack((box_segments, [r, th_rad, relevant_rs[0], relevant_ths[0], relevant_rs[1], relevant_ths[1]]))

        # OLD HEATMAP
        box_heatmap = np.zeros((360, ))
        for idx in range(box_segments.shape[0]):
            left = int(np.floor(np.degrees(box_segments[idx, 3])))
            right = int(np.floor(np.degrees(box_segments[idx, 5])))
            mid = (left + right) // 2
            #make a gaussian centered at mid with 3std at left, right
            std = (right - left) / 6
            for i in range(left, right + 1):
                poz = (i + 180) % 360
                val_gauss = np.exp(-((i - mid) ** 2) / (2 * std ** 2 + 0.001))
                box_heatmap[poz] = val_gauss if val_gauss > box_heatmap[poz] else box_heatmap[poz]

        box_heatmap = np.roll(box_heatmap, 180)

        if plots:
            plot_proc_data(nr_samps, cur_sample, sweep_points, sweep_frame, sweep_anns, box_corners, t0, nr_sweeps, max_range)

            plt.figure(figsize=(8, 8))
            plt.plot(np.arange(0, 360, 1), box_heatmap)
            plt.title('Sample: ' + str(nr_samps + 1) + ', nsweeps: ' + str(nr_sweeps) + ', Comp: ' + COMP_TYPE + ', Time: ' + str((cur_sample['timestamp'] - t0) / 1e6))


        cur_sample = nusc.get('sample', cur_sample['next'])

        if save_output:
            result[scene_name]['radar'].append(sweep_frame)
            result[scene_name]['anns'].append(sweep_anns)
            result[scene_name]['ego_vels'].append(cur_velocity)
            result[scene_name]['heatmap'].append(box_heatmap)

            # np.vstack((result[scene_name]['radar'], [sweep_frame]))
            # np.vstack((result[scene_name]['anns'], [sweep_anns]))
            # np.vstack((result[scene_name]['ego_vels'], [cur_velocity]))
            # np.vstack((result[scene_name]['heatmap'], [box_heatmap]))


            
            # print(sweep_anns)

    if save_output:
        # np.save(os.path.join(out_path, scene_name + '.npy'), result, allow_pickle=True)
        print(f'Elapsed time preproc {scene_name}:', timeit.default_timer() - start_time)

    return result

# %%

start_time = timeit.default_timer()
with Pool(32) as p:
    res = p.map(partial(preproc_scene, plots=False, save_output=True), range(len(nusc.scene)))
    # res = p.map(partial(preproc_scene, plots=False, save_output=True), range(32))
print('\n\nElapsed TOTAL time preproc dataset:', timeit.default_timer() - start_time)

start_time = timeit.default_timer()
final_res = reduce(lambda x, y: {**x, **y}, res, {})
np.save(os.path.join(OUT_PREPROC_PATH, 'all_scenes.npy'), final_res, allow_pickle=True)
print('Elapsed time reduce + save dataset:', timeit.default_timer() - start_time)


# %%

# res = preproc_scene(7, nr_samps=4, plots=True, max_range=50, nr_sweeps=6, ang_bins=360, save_output=False, min_radar_pts=1, min_lidar_pts=1, vis_level=0, only_vehicle=False, only_moving=False)

# render_whole_sample(nusc, nr_scene=0, nr_sample=2)

# print (len(res['0061']['radar']))

# render_whole_sample(nusc, nr_scene=1, nr_sample=2)
# %%

#DEBUG check how many points are used
# print('TOTAL POINTS:', total_radar_points, 'TOTAL USED POINTS:', total_used_radar_points)

#DEBUG check speed angles with respect to ego vehicle ./test_viteze
# np.save('/imec/other/dl4ms/nicule52/work/nuscenes_preproc/nuscenes-viz-pre/test_viteze/check_vit_poz_test.npy', check_vit_poz_test)

# TOTAL POINTS: 24330173 TOTAL USED POINTS: 16891177 -trainval whole, 6 sweep, 360bins, 60m, compensated velocities
