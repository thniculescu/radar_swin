# %%
import numpy as np
from pyquaternion import Quaternion
import yaml
from plot_utils import do_plots, get_ego_track_cart
import matplotlib.pyplot as plt

config_path = '../model_code/configs/alpha_small/all_targets.yaml'
# config_path = '../model_code/configs/alpha_small/only_veh.yaml'
# config_path = '../model_code/configs/alpha_small/no_static_only_veh.yaml'

with open(config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# data_path = config['DATA']['DATA_PATH']
data_path = '../preproc_datasets/abs_50m_1r_1l_0v_new/all_targets/all_scenes.npy'
print("data_path: ", data_path)
loaded_data = np.load(data_path, allow_pickle=True).item()


pred_data = np.load('../model_code/predictions.npy', allow_pickle=True).item()
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
SCENE_NAME = '0006'
# print(loaded_data.keys())

trans, rot = get_ego_track_cart(loaded_data, SCENE_NAME)

do_plots(loaded_data, SCENE_NAME, list(range(0, 6)), show_vels=False, ground_truth=True, predictions=True, vt=True, tracking=False, ego_track=(trans, rot), make_video=False)

# %%