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
from utils import preproc_scene
# %matplotlib inline

# %%

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
RadarPointCloud.dynprop_states = [0, 2, 6]
# RadarPointCloud.disable_filters()

nusc = NuScenes(version=DATASET_NAME, dataroot=DATASET_PATH, verbose=True)


# DEBUG check how many points are used
# total_radar_points = 0
# total_used_radar_points = 0

# %%

start_time = timeit.default_timer()
with Pool(16) as p:
    res = p.map(partial(preproc_scene, nusc, plots=False, save_output=True), range(len(nusc.scene)))

final_res = reduce(lambda x, y: {**x, **y}, res, {})
print('\n\nElapsed TOTAL time preproc dataset:', timeit.default_timer() - start_time)

start_time = timeit.default_timer()
np.save(os.join(OUT_PREPROC_PATH, 'all_scenes.npy'), final_res, allow_pickle=True)
print('Elapsed time save dataset:', timeit.default_timer() - start_time)

# %%
# start_time = timeit.default_timer()
# x = np.load('nuscenes_mare_preproc.npy', allow_pickle=True)
# print('Elapsed time load dataset:', timeit.default_timer() - start_time)


# %%

# res = preproc_scene(nusc, 0, nr_samps=50, plots=False, max_range=50, nr_sweeps=6, ang_bins=360, save_output=True, min_radar_pts=1, min_lidar_pts=0)

# render_whole_sample(nusc, nr_scene=0, nr_sample=2)

# print (len(res['0061']['radar']))

# render_whole_sample(nusc, nr_scene=1, nr_sample=2)
# %%

#DEBUG check how many points are used
# print('TOTAL POINTS:', total_radar_points, 'TOTAL USED POINTS:', total_used_radar_points)

#DEBUG check speed angles with respect to ego vehicle ./test_viteze
# np.save('/imec/other/dl4ms/nicule52/work/nuscenes_preproc/nuscenes-viz-pre/test_viteze/check_vit_poz_test.npy', check_vit_poz_test)

# TOTAL POINTS: 24330173 TOTAL USED POINTS: 16891177 -trainval whole, 6 sweep, 360bins, 60m, compensated velocities
