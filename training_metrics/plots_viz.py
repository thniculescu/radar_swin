# %%
import numpy as np
from pyquaternion import Quaternion
import yaml
from plot_utils import do_plots, gen_tracks, get_ego_track_cart
import matplotlib.pyplot as plt

# config_path = '../model_code/configs/alpha_small/all_targets_rot_tr.yaml'
# config_path = '../model_code/configs/alpha_small/only_veh.yaml'
# config_path = '../model_code/configs/alpha_small/no_static_only_veh.yaml'

# with open(config_path) as file:
#     config = yaml.load(file, Loader=yaml.FullLoader)

# data_path = config['DATA']['DATA_PATH']
# data_path = '../preproc_datasets/abs_50m_1r_1l_0v/no_static_only_veh/all_scenes.npy'


# data_path = '../preproc_datasets/abs_50m_1r_0l_0v_more/all_targets/all_scenes.npy'
# data_path = '../preproc_datasets/abs_50m_1r_0l_0v_more/no_static_only_veh/all_scenes.npy'
data_path = '../preproc_datasets/gt_all_anns/all_scenes.npy'


print("data_path: ", data_path)
loaded_data = np.load(data_path, allow_pickle=True).item()
train_scenes = set(loaded_data.keys())

# pred_path = 'predictions.npy'

pred_path = 'p_at.npy'
# pred_path = 'p_nsov.npy'
# pred_path = 'p_at_tr.npy'
# pred_path = 'p_nsov_tr.npy'

# pred_path = 'tr_new25_at.npy'
# pred_path = 'tr_new40_at.npy'
# pred_path = 'tr_new100_at.npy'

pred_data = np.load('../model_code/' + pred_path, allow_pickle=True).item()
print(pred_data['0006'].keys())
#delete from loaded data the scenes that are not in the predictions
for scene in list(loaded_data.keys()):
    if scene not in pred_data.keys():
        del loaded_data[scene]
    else:
        #merge the predictions dict with the loaded data dicts, use a oneliner
        loaded_data[scene].update(pred_data[scene])

# print("loaded_data keys: ", loaded_data.keys())
# print("loaded_data[scene_name] keys: ", loaded_data['0042'].keys())
# print("loaded_data[scene_name]['radar'] shape: ", len(loaded_data['0042']['radar']))
# print("loaded_data[scene_name]['anns'] shape: ", len(loaded_data['0042']['anns']))
# print("loaded_data[scene_name]['preds'] shape: ", len(loaded_data['0042']['preds']))
# %%
for scene_name in list(loaded_data.keys())[77:78]:
    print(scene_name)
    # continue
    trans, rot = get_ego_track_cart(loaded_data, scene_name)

    #DETECTION
    do_plots(loaded_data, scene_name, list(range(0, 4)), show_vels=False, ground_truth=True, predictions=True, vt=False, tracking=False, show_match_radius=None, ego_track=(trans, rot), make_video=False, add_legend=True)

    #TRACKING
    # do_plots(loaded_data, scene_name, list(range(0, 50)), show_vels=True, ground_truth=True, predictions=False, vt=True, tracking=True, show_match_radius=None, ego_track=(trans, rot), make_video=True, add_legend=True)

# %%
final_json = {
    "meta": {
        "use_camera" : False,
        "use_lidar" : False,
        "use_radar" : True,
        "use_map" : False,
        "use_external" : False,
    },
    "results": {},
}
    
for scene_name in list(loaded_data.keys()):
    ego_track = get_ego_track_cart(loaded_data, scene_name)
    track_ids, global_anns = gen_tracks(loaded_data[scene_name]['preds'], ego_track)

    for nr_sweep in range(len(track_ids)):
        sample_results = []
        ann_trans, ann_rot, ann_wl, ann_vel = global_anns[nr_sweep]

        sample_token = loaded_data[scene_name]['sample_token'][nr_sweep]
        
        for idx_ann in range(len(ann_trans)):
            sample_json = {}
            sample_json['sample_token'] = sample_token
            
            trans = ann_trans[idx_ann]
            rot = ann_rot[idx_ann]
            wl = ann_wl[idx_ann]
            vel = ann_vel[idx_ann]
            
            wl = np.hstack((wl, 2))

            rot0 = Quaternion(loaded_data[scene_name]['ego_rotation'][0]).rotation_matrix

            rot = rot0 @ rot
            vel = (rot0[:2, :2] @ vel.T).T

            #TODO: KEEPING ONLY MOVING PREDICTIONS
            if np.hypot(vel[0], vel[1]) < 0.1:
                continue
            
            trans = (rot0 @ trans.T).T
            trans += loaded_data[scene_name]['ego_translation'][0]
            trans[2] = 1

            sample_json['translation'] = trans.tolist()
            sample_json['rotation'] = Quaternion(matrix=rot).elements.tolist()
            sample_json['size'] = wl.tolist()
            sample_json['velocity'] = vel.tolist()

            sample_json['tracking_id'] = track_ids[nr_sweep][idx_ann]
            sample_json['tracking_name'] = 'car'
            sample_json['tracking_score'] = loaded_data[scene_name]['preds'][nr_sweep][idx_ann][7]
            assert loaded_data[scene_name]['preds'][nr_sweep][idx_ann][7] <= 1.0, "wrong heatmap conf"

            sample_json['detection_name'] = 'car'
            sample_json['detection_score'] = loaded_data[scene_name]['preds'][nr_sweep][idx_ann][7]
            sample_json['attribute_name'] = 'vehicle.moving'
            
            sample_results.append(sample_json)

        final_json['results'][sample_token] = sample_results

#print json to file
import json
from numpyencoder import NumpyEncoder

with open('results_' + pred_path[:-3] + 'json', 'w') as outfile:
    json.dump(final_json, outfile,
              indent=4, cls=NumpyEncoder)

# %%






        



#%% Calculate Splits
# for x in loaded_data.keys():
#     train_scenes.remove(x)

# test_scenes = list(loaded_data.keys())
# test_scenes = sorted(test_scenes)
# test_scenes = ['scene-' + x for x in test_scenes]

# train_scenes = list(train_scenes)
# train_scenes = sorted(train_scenes)
# train_scenes = ['scene-' + x for x in train_scenes]
# print(len(test_scenes))
# print(test_scenes)
# print(len(train_scenes))
# print(train_scenes)