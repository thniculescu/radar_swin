#%%

#each folder in ./plots is a scene
#make a video for each scene, concatenating all the frames in the folder

import cv2
import os
import imageio

def make_video(scene_folder):
    #get all the frames in the folder
    frames = [f for f in os.listdir(scene_folder) if os.path.isfile(os.path.join(scene_folder, f))]
    frames.sort(key=lambda x: int(x.split('.')[0]))
    #get the first frame to get the size
    # frame = cv2.imread(os.path.join(scene_folder, frames[0]))
    # height, width, layers = frame.shape
    # #create a video writer

    # video = cv2.VideoWriter(scene_folder + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 4, (width, height))
    #write each frame to the video
    
    #make video at 5 FPS
    # for frame in frames:
    #     video.write(cv2.imread(os.path.join(scene_folder, frame)))
    # video.release()

    ims = []
    for frame in frames:
        ims.append(imageio.imread(os.path.join(scene_folder, frame)))
    imageio.mimsave(scene_folder + '.gif', ims, duration=200)

#make a video for each scene
for scene in os.listdir('./plots'):
    if os.path.isdir(os.path.join('./plots', scene)):
        make_video(os.path.join('./plots', scene))

#save video to ./videos
#create ./videos if it doesn't exist
if not os.path.exists('./videos'):
    os.makedirs('./videos')

#move avi files to ./videos
for file in os.listdir('./plots'):
    #move all avi files to ./videos
    if file.endswith('.gif'):
        os.rename(os.path.join('./plots', file), os.path.join('./videos', file))