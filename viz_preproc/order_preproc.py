# %%
import os

data_path = '/imec/other/dl4ms/nicule52/nusc_preproc_data_viz2_sw2'

#for every folder in data_path
os.makedirs(os.path.join(data_path, 'radar'), exist_ok=True)
os.makedirs(os.path.join(data_path, 'anns'), exist_ok=True)
os.makedirs(os.path.join(data_path, 'hmap'), exist_ok=True)

for scene in os.listdir(data_path):
    if scene == 'radar' or scene == 'anns' or scene == 'hmap':
        continue
    print(scene)
    for radar_ann in os.listdir(os.path.join(data_path, scene)):
        for npdata in os.listdir(os.path.join(data_path, scene, radar_ann)):
            # print(npdata)
            os.rename(os.path.join(data_path, scene, radar_ann, npdata), os.path.join(data_path, radar_ann, "scene_" + scene.split('-')[1] + '_sweep_' + npdata.split('_')[0] + '.npy'))