# %%
import numpy as np
import os
import matplotlib.pyplot as plt

SCENE = 'scene_0001'
NRSWEEPS = 6
NR_SAMP = 1
BASE_PATH = '/imec/other/dl4ms/nicule52/nusc_preproc_data/train'
# BASE_PATH = '/imec/other/dl4ms/nicule52/preproc_datasets/nusc_preproc_data/train' # unfiltered anns


anns = np.load(os.path.join(BASE_PATH, 'anns', f'{SCENE}_sweep_{NR_SAMP}.npy'))

print(anns.shape)
#make polar subplot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
ax.scatter(anns[:, 1], anns[:, 0], marker='x', c='r', s=30)


radar = np.load(os.path.join(BASE_PATH, 'radar', f'{SCENE}_sweep_{NR_SAMP}.npy'))

print(radar.shape)
#make polar subplot
for idx in range(radar.shape[0]):
    ax.scatter(np.radians(np.arange(360)), radar[idx, :, 0], marker='o', c='b', s=3)

ax.set_ylim(0, 60)

try:
    hmap = np.load(os.path.join(BASE_PATH, 'hmap', f'{SCENE}_sweep_{NR_SAMP}.npy'))

    print(hmap.shape)

    fig, ax = plt.subplots()
    ax.plot(np.arange(360), hmap)
except:
    print('No heatmap available')