import os
import timeit
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix, view_points
# from nuscenes import render_sample

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

ANN_FEATURE_SIZE = 7 + 4 + 5
#  range [m], angle [rad], orientation@angle [rad], size/r [w l], velocity/r [vr vt]
#  segment-genheatmap [r_min th_min r_max th_max],
#  visibility [0-4], 
#  class
#  attribute(moving status)
#  num_radar_pts, num_lidar_pts
COMP_TYPE = 'COMP' # COMP, RAW, COMP_MAN, RAW_MAN

def get_points(nusc, cur_sample, chan_type, chan_tok, ref_chan, ref_sd_record, nsweeps, speed_ref, max_range=60, ego_vels=None):
    pc, times = RadarPointCloud.from_file_multisweep(nusc, cur_sample, chan_type, ref_chan, nsweeps=nsweeps, min_distance=1.0)
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
    speeds = np.hypot(deltas_vel[:, 0], deltas_vel[:, 1]) * np.sign(np.cos(np.arctan2(deltas_vel[:, 1], deltas_vel[:, 0]) - np.arctan2(ys, xs)))
    # speeds = np.hypot(velocities[0], velocities[1])
    # speeds = np.hypot(velocities[0], velocities[1]) * np.sign(np.cos(np.arctan2(ys, xs) - np.arctan2(velocities[1], velocities[0])))
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
    binned_sweep = np.full((nr_ang_bins, 3), 0, dtype=np.float32)
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


def filter_boxes(nusc, anns, max_range=60, vis_level=0, min_radar_pts=1, min_lidar_pts=1, only_vehicle=True, only_moving_targets=False, skip_filter=False):
    if skip_filter:
        return anns
    filtered_anns = []
    for ann in anns:
        sample_ann = nusc.get('sample_annotation', ann)
        # pprint.pprint(sample_ann)
        if only_vehicle and not sample_ann["category_name"].startswith('vehicle'):
            continue

        if len(sample_ann['attribute_tokens']) > 0:
            ann_att = nusc.get('attribute', sample_ann['attribute_tokens'][0])['name']
        else:
            ann_att = 'other_obj.static'

        if only_moving_targets and \
            ann_att != 'vehicle.moving' and \
            ann_att != 'cycle.with_rider' and \
            ann_att != 'pedestrian.moving':
            continue

        if int(sample_ann["visibility_token"]) < vis_level:
            continue
        if int(sample_ann["num_radar_pts"]) < min_radar_pts:
            continue
        if int(sample_ann["num_lidar_pts"]) < min_lidar_pts:
            continue
        #TODO filter boxes further than max_range

        filtered_anns.append(ann)

    return filtered_anns


def plot_proc_data(nr_samps, cur_sample, sweep_points, sweep_frame, sweep_anns, box_corners, t0, nr_sweeps, max_range=60, make_movie=False, scene_name=None):
    fig, ax = plt.subplots(subplot_kw={'polar': 'True'}, figsize=(8, 8))
    #set maxrange 60
    ax.set_title('Sample: ' + str(nr_samps + 1) + ' Time: ' + str((cur_sample['timestamp'] - t0) / 1e6))
    ax.set_ylim(0, max_range)
    # ax.set_yticks(np.arange(0, max_range, 1))

    colors = ['#00FF00', '#00D815', '#00B13C', '#008A63', '#00638A', '#003CB1', '#0015D8', '#0000FF'][::-1]

    for idx in reversed(range(nr_sweeps)):
        
        # angles = np.radians(np.arange(0, sweep_frame.shape[1], 1))
        angles = np.linspace(0, 2 * np.pi, sweep_frame.shape[1], endpoint=False)

        #points
        ax.scatter(angles, 
                   sweep_frame[idx][:, 0], 
                   marker='o', s=7, c=colors[idx], 
                   alpha=1 - 0.9 * idx / nr_sweeps)

        #velocities
        # ax.quiver(angles, 
        #           sweep_frame[idx][:, 0], 
        #           np.zeros_like(angles),
        #           sweep_frame[idx][:, 1],
        #           angles='xy', scale_units='xy', scale=1/6,
        #           color='k', 
        #           alpha=1 - 0.9 * idx / nr_sweeps)
        
        #not binned points
        # ax.scatter(np.radians(sweep_points[idx][:, 1]), sweep_points[idx][:, 0], s=1, c='r')

    
    ax.scatter(sweep_anns[:, 1], sweep_anns[:, 0], s=25, c='r', marker='x')
        
    for idx in range(sweep_anns.shape[0]):
        cor = box_corners[idx]
        cor = np.vstack((cor, cor[0, :]))
        
        #tranform to polar
        ranges = np.hypot(cor[:, 0], cor[:, 1])
        angles = np.arctan2(cor[:, 1], cor[:, 0])
        ax.plot(angles, ranges, 'r')

        color = 'r'
        #plot radial velocity
        ax.arrow(sweep_anns[idx, 1], sweep_anns[idx, 0], 0, sweep_anns[idx, 0] * sweep_anns[idx, 5], lw=4, fc=color, ec=color, head_width=0, head_length=0, alpha=0.5)

        #plot tangential velocity
        ranges = [sweep_anns[idx, 0], np.hypot(sweep_anns[idx, 0], sweep_anns[idx, 6] * sweep_anns[idx, 0])]
        angles = [sweep_anns[idx, 1], sweep_anns[idx, 1] + np.arcsin(sweep_anns[idx, 6] * sweep_anns[idx, 0] / ranges[1])]       
        ax.arrow(angles[0], ranges[0], angles[1] - angles[0], ranges[1] - ranges[0], lw=4, fc=color, ec=color, head_width=0, head_length=0, alpha=0.5)



        # OLD plot radial velocity
        r, th_ang = sweep_anns[idx, :2]
        ranges = [r, r + sweep_anns[idx, 5] * r]
        angles = [th_ang, th_ang]

        if ranges[1] < 0:
            ranges[1] = -ranges[1]
            angles[1] = angles[1] + np.pi
        ax.plot(angles, ranges, 'g')

        #OLD plot tangential velocity
        ranges = [r, np.hypot(r, sweep_anns[idx, 6] * r)]
        angles = [th_ang, th_ang + np.arcsin(sweep_anns[idx, 6] * r / ranges[1])]
        ax.plot(angles, ranges, 'g')
        
    
    if make_movie:
        os.makedirs(f'./plots/{scene_name}', exist_ok=True)
        plt.savefig(f'./plots/{scene_name}/' + str(nr_samps) + '.png')
        plt.close()


def render_whole_sample(nusc: 'NuScenes', nr_scene, nr_sample, nr_sweeps=6):
    first = nusc.scene[nr_scene]['first_sample_token']
    while nr_sample > 0:
        cur_sample = nusc.get('sample', first)
        first = cur_sample['next']
        nr_sample -= 1

    nusc.render_sample(first, nsweeps=nr_sweeps, verbose=False)



#################################################################################################################
