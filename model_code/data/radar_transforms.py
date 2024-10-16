import torch

class input_transform(object):

    def __init__(self, config):
        self.config = config
    
    def __call__(self, sample):
        sample[sample == 1e6] = 0 #TODO: making empty bins 0
        # assert torch.any(torch.isnan(sample)) == False, 'NaN input radar data'
        return sample.permute(2, 0, 1).to(torch.float16)
    

class target_transform(object):

    def __init__(self, config):
        self.config = config
    
    def __call__(self, target):
        anns, hmap = target
        # ground truth: ANN_FEATURE_SIZE = 7
        # range - m
        # angle - radians
        # orientation - front relative to ray - radians
        # size - [w l] - relative to range
        # velocity - [vr vt] - relative to range

        # heatmap: (360,)

        hmap = hmap.to(torch.float16).reshape(-1, 1)
        anns = anns.to(torch.float16)

        anns[torch.isnan(anns)] = 0
        # assert torch.any(torch.isnan(anns)) == False, 'NaN anns ref data'
        # assert torch.any(torch.isnan(hmap)) == False, 'NaN heatmap ref data'
        # print(anns[0])

        anns[:, [0, 1]] = anns[:, [1, 0]] #angle on first position
        
        #round the angle to the nearest degree - range bins
        anns[:, 0] = torch.floor(torch.rad2deg(anns[:, 0]))

        anns = torch.hstack([anns[:, :3], anns[:, [2]], anns[:, 3:7]]) # double orientation column
        anns[:, 2] = torch.sin(anns[:, 2])
        anns[:, 3] = torch.cos(anns[:, 3])

        #create a new tensor called new_anns (360, 6), place the anns[1:] in the tensor at the index of anns[1]
        new_anns = torch.zeros((360, 7), dtype=torch.float16)
        new_anns[anns[:, 0].int()] = anns[:, 1:]
        new_anns = new_anns.to(torch.float16)
        mask = torch.zeros((360, 1), dtype=torch.float16)
        mask[anns[:, 0].int()] = 1

        # print(anns.shape)
        # print(torch.sum(mask))

        # print(new_anns[mask[:, 0] == 1])

        target = torch.hstack([hmap, new_anns, mask]) # 360 x 8 + 1 : hmap, range, orientation [sin, cos], size [w l], velocity [vr vt], mask

        return target