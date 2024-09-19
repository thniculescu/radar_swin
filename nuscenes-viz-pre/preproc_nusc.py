# %%
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import pprint
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix, view_points
import os

# %matplotlib inline
# %%

CATEGORY_NAMES = {
    'animal': 0,
    'human.pedestrian.adult': 1,
    'human.pedestrian.child': 2,
    'human.pedestrian.construction_worker': 3,
    'human.pedestrian.personal_mobility': 4,
    'human.pedestrian.police_officer': 5,
    'human.pedestrian.stroller': 6,
    'human.pedestrian.wheelchair': 7,
    'movable_object.barrier': 8,
    'movable_object.debris': 9,
    'movable_object.pushable_pullable': 10,
    'movable_object.trafficcone': 11,
    'static_object.bicycle_rack': 12,
    'vehicle.bicycle': 13,
    'vehicle.bus.bendy': 14,
    'vehicle.bus.rigid': 15,
    'vehicle.car': 16,
    'vehicle.construction': 17,
    'vehicle.emergency.ambulance': 18,
    'vehicle.emergency.police': 19,
    'vehicle.motorcycle': 20,
    'vehicle.trailer': 21,
    'vehicle.truck': 22,
}

ATTRIBUTE_NAMES = {
    'vehicle.moving': 0,
    'vehicle.stopped': 1,
    'vehicle.parked': 2,
    'cycle.with_rider': 3,
    'cycle.without_rider': 4,
    'pedestrian.sitting_lying_down': 5,
    'pedestrian.standing': 6,
    'pedestrian.moving': 7,
    'other_obj.static': 8, #added for other objects which are without attribute (movable_obj, static_obj, animal)
}

# %%

def get_points(nusc, cur_sample, chan_type, chan_tok, ref_chan, ref_sd_record, nsweeps, speed_ref, max_range=60, ego_vels=None):
    pc, times = RadarPointCloud.from_file_multisweep(nusc, cur_sample, chan_type, ref_chan, nsweeps=nsweeps)
    #use global timestamps    
    ref_time = 1e-6 * ref_sd_record['timestamp']
    times = ref_time - times #use global timestamps
    times = np.round(times, 3)
    times = times.flatten()


    # find global ego_vel at each timestamp
    time_ego_vels_indices = [np.argmin(np.abs(ego_vels[0][:] - t)) for t in times]
    time_ego_vels = ego_vels[1][time_ego_vels_indices]
    time_ego_vels = np.array(time_ego_vels)
    time_ego_vels = time_ego_vels[:, :2]
    # print('TIME EGO VELS:', time_ego_vels.shape)
    # print('TIMES:', times.shape)
    # print('pc', pc.points.shape)

    # prepare time_ego_vels for subtraction with velocities
    time_ego_vels = time_ego_vels.T
    time_ego_vels = np.vstack((time_ego_vels, np.zeros(time_ego_vels.shape[1])))


    # Transform radar velocities (x is front, y is left), as these are not transformed when loading the
    # point cloud.
    sd_record = nusc.get('sample_data', chan_tok)
    radar_cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    ref_cs_record = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
    cs_record = nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
    ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                    rotation=Quaternion(cs_record["rotation"]))
    # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
    ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
    rotation_vehicle_flat_from_vehicle = np.dot(
        Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
        Quaternion(pose_record['rotation']).inverse.rotation_matrix)
    vehicle_flat_from_vehicle = np.eye(4)
    vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
    viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)

    # Transform radar points to ego vehicle frame
    my_pc = view_points(pc.points[:3, :], viewpoint, normalize=False)
    my_pc = my_pc.T

    if speed_ref == 'COMP' or speed_ref == 'RAW_MAN':
        velocities = pc.points[8:10, :]  # Compensated velocity
        velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
        velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
        velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
    elif speed_ref == 'RAW' or speed_ref == 'COMP_MAN':
        velocities = pc.points[6:8, :]  # Raw velocity
        velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))

        # first one extra
        
        # velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
        velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
    else:
        raise ValueError("Error: Invalid speed_ref")
    

    rot_time_ego_vels = np.dot(Quaternion(pose_record['rotation']).rotation_matrix.T, time_ego_vels)
    rot_time_ego_vels = np.vstack((rot_time_ego_vels[0], rot_time_ego_vels[1], np.zeros(rot_time_ego_vels.shape[1])))
    rot_time_ego_vels = rot_time_ego_vels.T
    if speed_ref == 'COMP_MAN' or speed_ref == 'RAW_MAN':
        rot_time_ego_vels = np.dot(Quaternion(pose_record['rotation']).rotation_matrix.T, time_ego_vels)
        rot_time_ego_vels = np.vstack((rot_time_ego_vels[0], rot_time_ego_vels[1], np.zeros(rot_time_ego_vels.shape[1])))
        rot_time_ego_vels = rot_time_ego_vels.T

        #project ego velocities on lines 0---my_pc[:, 0], my_pc[:, 1]
        proj_time_ego_vels = rot_time_ego_vels
        proj_time_ego_vels = ((proj_time_ego_vels[:, 0:2] * my_pc[:, 0:2]).sum(1) / (my_pc[:, 0:2] * my_pc[:, 0:2]).sum(1))
        proj_time_ego_vels = np.vstack((proj_time_ego_vels, proj_time_ego_vels)).T
        proj_time_ego_vels = proj_time_ego_vels *  my_pc[:, 0:2]
        proj_time_ego_vels = np.vstack((proj_time_ego_vels.T, np.zeros(proj_time_ego_vels.shape[0])))

    velocities[2, :] = np.zeros(pc.points.shape[1])
    my_pc = np.hstack((my_pc, velocities.T))

    points_vel = view_points(pc.points[:3, :] + my_pc[:, 3:6].T, viewpoint, normalize=False).T
    deltas_vel = points_vel - my_pc[:, :3]


    if speed_ref == 'RAW' or speed_ref == 'COMP_MAN':
        v_rad_scaling = deltas_vel[:, 0] / np.hypot(my_pc[:, 0], my_pc[:, 1])
        deltas_vel[:, 0] = my_pc[:, 0] * v_rad_scaling 
        deltas_vel[:, 1] = my_pc[:, 1] * v_rad_scaling

        # deltas_vel -= rot_time_ego_vels
        
        # deltas_vel = ((deltas_vel[:, 0:2] * my_pc[:, 0:2]).sum(1) / (my_pc[:, 0:2] * my_pc[:, 0:2]).sum(1))
        # deltas_vel = np.vstack((deltas_vel, deltas_vel)).T
        # deltas_vel = deltas_vel *  my_pc[:, 0:2]
        # deltas_vel = np.vstack((deltas_vel.T, np.zeros(deltas_vel.shape[0])))
        # deltas_vel = deltas_vel.T

        deltas_vel[:, 2] = 0

    if speed_ref == 'COMP_MAN':
        deltas_vel += proj_time_ego_vels.T
    elif speed_ref == 'RAW_MAN':
        deltas_vel -= proj_time_ego_vels.T

    

    #RUINS VELOCITY ORIENTATION: nusc official plot clip
    # deltas_vel = 6 * deltas_vel  # Arbitrary scaling
    # max_delta = 20
    # deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)

    my_pc = np.hstack((my_pc, deltas_vel))


    #TODO wat do
    # if my_pc.shape[0] == 0:
    #     continue

    
    # print(my_pc.shape)
    #TODO UPDATE POINTSSSSSS based on my_pc delta vel
    xs = my_pc[:, 0]
    ys = my_pc[:, 1]

    #convert to range and angle in 360 degrees
    ranges = np.hypot(xs, ys)
    angles = np.degrees(np.arctan2(ys, xs))
    angles[angles < 0] += 360
    speeds = np.hypot(velocities[0], velocities[1])
    rcs = pc.points[5, :].T
    
    # mask angles, only when range is < max_range
    mask = ranges < max_range
    ranges = ranges[mask]
    angles = angles[mask]
    speeds = speeds[mask]
    rcs = rcs[mask]
    times = times.flatten()[mask]
    # times = (times - t0) / 1e6

    points = np.vstack((ranges, angles, speeds, rcs)).T

    return points, times, my_pc


def split_time(points, times):

    points = np.append(points, times[:, None], axis=1)

    times = np.unique(times)
    times = np.sort(times)[::-1]

    new_points = []

    for t in times:
        #split points by time
        mask = points[:, -1] == t
        # print(mask)
        # print(points[mask].shape)
        # print(points[mask])
        new_points.append(points[mask])

    return new_points, times


def split_ang_bins(points, nr_ang_bins=360): # points = [range, angle, speed, rcs, time]


    #round down the angle
    points[:, 1] = np.floor(points[:, 1])
    #sort by range
    points = points[np.argsort(points[:, 0])]
    #unique by angle, keep first
    _, idx = np.unique(points[:, 1], return_index=True)
    points = points[idx]

    # binned_sweep = np.zeros((nr_ang_bins, 3))
    binned_sweep = np.full((nr_ang_bins, 3), 0)
    for p in points:
        binned_sweep[int(p[1]), 0] = p[0]
        binned_sweep[int(p[1]), 1] = p[2]
        binned_sweep[int(p[1]), 2] = p[3]

    return binned_sweep, idx.shape[0]


def calc_ego_vel(nusc, cur_sample):
    ego_poses_list = []
    for chan_type, chan_tok in cur_sample['data'].items():
        if 'RADAR' in chan_type:
            sweep_data = nusc.get('sample_data', chan_tok)
            while sweep_data['next'] != '':
                sweep_data = nusc.get('sample_data', sweep_data['next'])
                ego_data = nusc.get('ego_pose', sweep_data['ego_pose_token'])
                ego_poses_list.append((ego_data['timestamp'], ego_data['translation'], ego_data['rotation']))

    ego_vel_list = []
    for i in range(1, len(ego_poses_list) - 1):
        dt = (ego_poses_list[i + 1][0] - ego_poses_list[i - 1][0]) / 1e6
        vxy = np.array(ego_poses_list[i + 1][1]) - np.array(ego_poses_list[i - 1][1])
        ego_vel_list.append((np.round(ego_poses_list[i][0] / 1e6, 3), vxy / dt))

    ego_vel_list = sorted(ego_vel_list, key=lambda x: x[0])
    #unzip ego vel list
    ego_vel_list = list(zip(*ego_vel_list))

    timestamps = np.array(ego_vel_list[0])
    ego_vel_trans = np.array(ego_vel_list[1])

    return timestamps, ego_vel_trans

def filter_boxes(nusc, anns, max_range=60, vis_level=4, min_radar_pts=1, only_vehicle=True, skip_filter=False):
    if skip_filter:
        return anns
    filtered_anns = []
    for ann in anns:
        sample_ann = nusc.get('sample_annotation', ann)
        # pprint.pprint(sample_ann)
        if only_vehicle and not sample_ann["category_name"].startswith('vehicle'):
            continue
        if int(sample_ann["visibility_token"]) < vis_level:
            continue
        # if int(sample_ann["num_radar_pts"]) < num_radar_pts:
        #     continue
        #TODO filter boxes further than max_range

        filtered_anns.append(ann)

    return filtered_anns
# %%

PLOTS = True
MAX_RANGE = 60
NR_SWEEPS = 2
RADAR_CHANS = {"RADAR_FRONT" : 0, "RADAR_FRONT_LEFT" : 1, "RADAR_FRONT_RIGHT" : 2, "RADAR_BACK_LEFT" : 3, "RADAR_BACK_RIGHT" : 4}
ANG_BINS = 360
ANN_FEATURE_SIZE = 7 + 4 + 1 + 1 + 1 # range [m], angle [rad], orientation@angle [rad], size/r [w l], velocity/r [vr vt], segment-genheatmap [r_min th_min r_max th_max], visibility [0-4], class, attribute(moving status)
COMP_TYPE = 'RAW_MAN' # COMP, RAW, COMP_MAN, RAW_MAN
RadarPointCloud.default_filters()
# RadarPointCloud.disable_filters()
# nusc = NuScenes(version='v1.0-trainval', dataroot='/imec/other/dl4ms/nicule52/datasets/nuscenes/full-v1.0/', verbose=True)
nusc = NuScenes(version='v1.0-mini', dataroot='/imec/other/dl4ms/nicule52/datasets/nuscenes/v1.0-mini/', verbose=True)
# %%
nusc.list_scenes()
check_vit_poz_test = np.empty((0, 4))
check_vit_poz_test2 = np.empty((0, 4))


total_radar_points = 0
total_used_radar_points = 0
os.makedirs('preproc_data/radar', exist_ok=True)
os.makedirs('preproc_data/anns',  exist_ok=True)
os.makedirs('preproc_data/ego_vels',  exist_ok=True)

# for scene_id in range(1, 2):
for scene_id in range(5, 6):
# for scene_id in range(2, 3):
# for scene_id in range(6, 7):
# for scene_id in range(4, 5):
# for scene_id in range(len(nusc.scene)):
    print('Scene:', scene_id)
    # if int(nusc.scene[scene_id]['name'].split('-')[1]) < 1010:
    #     continue
    print(nusc.scene[scene_id]['name'])
    print('num_samps:', nusc.scene[scene_id]['nbr_samples'])
    cur_sample = nusc.get('sample', nusc.scene[scene_id]['first_sample_token'])
    global_ego_vel = calc_ego_vel(nusc, cur_sample)

    cur_sample = nusc.get('sample', cur_sample['next'])
    t0 = cur_sample['timestamp']
    
    for nr_samps in range(1):
    # for nr_samps in range(nusc.scene[scene_id]['nbr_samples'] - 2):
        # if nr_samps not in range(7, 8):
        #     cur_sample = nusc.get('sample', cur_sample['next'])
        #     continue

        # print('Sample:', nr_samps + 1)
        
        samp_sweep_nr_points = 0
        ref_chan = 'LIDAR_TOP'
        ref_sd_record = nusc.get('sample_data', cur_sample['data'][ref_chan])

        sweep_points = [np.empty((0, 5)) for i in range(NR_SWEEPS)] # for each "sweep", concatenate the points from diff channels, t -> t - NR_SWEEPS + 1
        sweep_frame = np.empty((NR_SWEEPS, ANG_BINS, 3)) #range, speed, rcs

        # if PLOTS:
        #     fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        #     ax.axis('on')
        #     ax.grid()
        #     ax.set_xticks(np.arange(-MAX_RANGE, MAX_RANGE, 10))
        #     ax.set_yticks(np.arange(-MAX_RANGE, MAX_RANGE, 10))

        #     gigel = np.arange(0, 360, 2)
        #     gigel = np.radians(gigel)
        #     for gogu in range(1, 7):
        #         ax.plot(np.cos(gigel) * gogu * 10, np.sin(gigel) * gogu * 10, 'k--', linewidth=0.5)

        
        # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        # ax.axis('on')
        # ax.grid()
        # ax.set_xticks(np.arange(-MAX_RANGE, MAX_RANGE, 10))
        # ax.set_yticks(np.arange(-MAX_RANGE, MAX_RANGE, 10))
        # ax.set_xlim(-MAX_RANGE, MAX_RANGE)
        # ax.set_ylim(-MAX_RANGE, MAX_RANGE)
        for chan_type, chan_tok in cur_sample['data'].items():
            if 'RADAR' in chan_type:
                # if not (chan_type == 'RADAR_FRONT_LEFT' or chan_type == 'RADAR_FRONT'):
                # if not (chan_type == 'RADAR_FRONT_LEFT'):
                #     continue

                # if PLOTS:
                #     # nusc.render_sample_data(chan_tok, ax=ax, nsweeps=NR_SWEEPS, axes_limit=MAX_RANGE, underlay_map=True, with_anns=True, verbose=False)
                #     ax.axis('on')
                #     ax.grid()
                #     ax.set_xticks(np.arange(-MAX_RANGE, MAX_RANGE, 10))
                #     ax.set_yticks(np.arange(-MAX_RANGE, MAX_RANGE, 10))
                #     ax.set_xlim(-MAX_RANGE, MAX_RANGE)
                #     # ax.set_xlim(-50, -20)
                #     ax.set_ylim(-MAX_RANGE, MAX_RANGE)
                #     # ax.set_ylim(-10, 20)
                #     ax.set_xlabel('x')
                #     ax.set_ylabel('y')
                #     ax.set_title('Sample: ' + str(nr_samps + 1) + ', nsweeps: ' + str(NR_SWEEPS) + ', Comp: ' + COMP_TYPE + ', Time: ' + str((cur_sample['timestamp'] - t0) / 1e6))


                points, times, test_pc = get_points(nusc, cur_sample, chan_type, chan_tok, ref_chan, ref_sd_record, nsweeps=NR_SWEEPS, speed_ref='COMP', max_range=MAX_RANGE, ego_vels=global_ego_vel)
                # points, times, test_pc2 = get_points(nusc, cur_sample, chan_type, chan_tok, ref_chan, ref_sd_record, nsweeps=NR_SWEEPS, speed_ref='RAW_MAN', max_range=MAX_RANGE, ego_vels=global_ego_vel)
                # print('CHAN:', chan_type, "NR_SAMP: ", nr_samps + 1, "TIME:", (cur_sample['timestamp'] - t0) / 1e6)
                # print('SHAPES:', points.shape, times.shape)
                check_vit_poz_test = np.append(check_vit_poz_test, test_pc[:, [0, 1, 6, 7]], axis=0)
                # check_vit_poz_test2 = np.append(check_vit_poz_test2, test_pc2[:, [0, 1, 6, 7]], axis=0)

                
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



                
        # if PLOTS:
        #     if not os.path.exists('plots/' + nusc.scene[scene_id]['name']):
        #         os.makedirs('plots/' + nusc.scene[scene_id]['name'])
        #     fig.savefig('plots/' + nusc.scene[scene_id]['name'] + '/' + str(nr_samps) + '_nrsweeps' + str(NR_SWEEPS) + '.png')
        #     plt.close(fig)
        

        for sw, p in enumerate(sweep_points):
            binned_points, used_points = split_ang_bins(p, nr_ang_bins=ANG_BINS)
            # print('ALLL POINTS:', p.shape[0], 'USED POINTS:', used_points)
            total_radar_points += p.shape[0]
            total_used_radar_points += used_points
            sweep_frame[sw] = binned_points
        

        # filtered_anns = cur_sample['anns']
        filtered_anns = filter_boxes(nusc, cur_sample['anns'], vis_level=0, skip_filter=True)

        # go through sample, get all annotation positions and box sizes
        _, boxes, _ = nusc.get_sample_data(cur_sample['data'][ref_chan], selected_anntokens=filtered_anns, use_flat_vehicle_coordinates=True)
        sweep_anns = np.empty((0, ANN_FEATURE_SIZE))

        # print('filt anns:', len(filtered_anns))
        # print('boxes:', len(boxes))


        cur_yaw = Quaternion(nusc.get('ego_pose', ref_sd_record['ego_pose_token'])['rotation']).yaw_pitch_roll[0]
        cur_velocity = global_ego_vel[1][np.argmin(np.abs(global_ego_vel[0] - cur_sample['timestamp'] / 1e6))]
        cur_velocity = np.dot(Quaternion(scalar=np.cos(cur_yaw / 2), vector=[0, 0, np.sin(cur_yaw / 2)]).inverse.rotation_matrix, cur_velocity)
        cur_velocity = cur_velocity[:2]

        box_corners = np.empty((0, 4, 2))

        for box, ann in zip(boxes, filtered_anns):

            box.velocity = nusc.box_velocity(ann)
            box.velocity = np.nan_to_num(box.velocity)

            r = np.hypot(box.center[0], box.center[1])

            if r > MAX_RANGE:
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

            # ground truth: ANN_FEATURE_SIZE = 14
            # range - m
            # angle - radians
            # orientation - front relative to ray - radians
            # size - [w l] - relative to range
            # velocity - [vr vt] - relative to range
            # segment - [r_min th_min r_max th_max]
            # visibility - 0-4
            # class - 0-22
            # attribute - 0-8

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

            # box_segments = np.vstack((box_segments, [r, th_rad, relevant_rs[0], relevant_ths[0], relevant_rs[1], relevant_ths[1]]))
            
            sample_ann = nusc.get('sample_annotation', ann)
            box_vis = int(sample_ann['visibility_token'])
            box_class = CATEGORY_NAMES[sample_ann['category_name']]

            if len(sample_ann['attribute_tokens']) > 0:
                ann_att = nusc.get('attribute', sample_ann['attribute_tokens'][0])['name']
            else:
                ann_att = 'other_obj.static'

            box_att = ATTRIBUTE_NAMES[ann_att] 

            sweep_anns = np.vstack((sweep_anns,
                np.hstack((
                    r, th_rad, rel_orient_rad, *rel_size, *vel,
                    relevant_rs[0], relevant_ths[0], relevant_rs[1], relevant_ths[1],
                    box_vis, box_class, box_att))))

            #TODO: segment discontinuity
            # if relevant_ths[0] < np.pi and relevant_ths[1] > np.pi:
            #     new_r = relevant_rs[0] + (relevant_rs[1] - relevant_rs[0]) * (np.pi - relevant_ths[0]) / (relevant_ths[1] - relevant_ths[0])
            #     box_segments = np.vstack((box_segments, [r, th_rad, relevant_rs[0], relevant_ths[0], relevant_rs[1], relevant_ths[1]]))
            # else:
            #     box_segments = np.vstack((box_segments, [r, th_rad, relevant_rs[0], relevant_ths[0], relevant_rs[1], relevant_ths[1]]))

        # OLD HEATMAP
        # box_heatmap = np.zeros((360, ))
        # for idx in range(box_segments.shape[0]):
        #     left = int(np.floor(np.degrees(box_segments[idx, 3])))
        #     right = int(np.floor(np.degrees(box_segments[idx, 5])))
        #     mid = (left + right) // 2
        #     #make a gaussian centered at mid with 3std at left, right
        #     std = (right - left) / 6
        #     for i in range(left, right + 1):
        #         poz = (i + 180) % 360
        #         val_gauss = np.exp(-((i - mid) ** 2) / (2 * std ** 2 + 0.001))
        #         box_heatmap[poz] = val_gauss if val_gauss > box_heatmap[poz] else box_heatmap[poz]

        # box_heatmap = np.roll(box_heatmap, 180)

        # if PLOTS:
        #     fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
        #     #set maxrange 60
        #     ax.set_title('Sample: ' + str(nr_samps + 1) + ' Time: ' + str((cur_sample['timestamp'] - t0) / 1e6))
        #     ax.set_ylim(0, MAX_RANGE)
        #     for sw in reversed(range(NR_SWEEPS)):
        #         # ax.scatter(np.radians(sweep_points[sw][:, 1]), sweep_points[sw][:, 0], s=7, c='r')

        #         ax.scatter(np.radians(np.arange(0, 360, 1)), sweep_frame[sw][:, 0], marker='o', s=4, c='b')

            
        #     ax.scatter(sweep_anns[:, 1], sweep_anns[:, 0], s=47, c='r', marker='x')
                
        #     for idx in range(sweep_anns.shape[0]):
        #         cor = box_corners[idx]
        #         cor = np.vstack((cor, cor[0, :]))

        #         r, th_ang = sweep_anns[idx, :2]
                
        #         #tranform to polar
        #         ranges = np.hypot(cor[:, 0], cor[:, 1])
        #         angles = np.arctan2(cor[:, 1], cor[:, 0])
        #         ax.plot(angles, ranges, 'r')

        #         #plot radial velocity
        #         ranges = [r, r + sweep_anns[idx, 5] * r]
        #         angles = [th_ang, th_ang]

        #         if ranges[1] < 0:
        #             ranges[1] = -ranges[1]
        #             angles[1] = angles[1] + np.pi
        #         ax.plot(angles, ranges, 'k')

        #         #plot tangential velocity
        #         ranges = [r, np.hypot(r, sweep_anns[idx, 6] * r)]
        #         angles = [th_ang, th_ang + np.arcsin(sweep_anns[idx, 6] * r / ranges[1])]
        #         ax.plot(angles, ranges, 'k')

        np.save('preproc_data/' + 'radar/' + 'scene_' + nusc.scene[scene_id]['name'].split('-')[1] + "_sweep_"  + str(nr_samps + 1) + '.npy', sweep_frame)
        np.save('preproc_data/' + 'anns/'  + 'scene_' + nusc.scene[scene_id]['name'].split('-')[1] + "_sweep_"  + str(nr_samps + 1) + '.npy', sweep_anns)
        np.save('preproc_data/' + 'ego_vels/'  + 'scene_' + nusc.scene[scene_id]['name'].split('-')[1] + "_sweep_"  + str(nr_samps + 1) + '.npy', cur_velocity)
        cur_sample = nusc.get('sample', cur_sample['next'])



print('TOTAL POINTS:', total_radar_points, 'TOTAL USED POINTS:', total_used_radar_points)
np.save('/imec/other/dl4ms/nicule52/nuscenes-viz-pre/test_viteze/check_vit_poz_test.npy', check_vit_poz_test)
# np.save('check_vit_poz_test2.npy', check_vit_poz_test2)


# %%
# TOTAL POINTS: 24330173 TOTAL USED POINTS: 16891177 -trainval whole, 6 sweep, 360bins, 60m, compensated velocities
