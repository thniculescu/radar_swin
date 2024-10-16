import torch.nn as nn
import torch

class RadarSwinLoss(nn.Module):

    def __init__(self, config, range_es=1.0, bbox_size_ori=1.0, velocity=1.0):
        super(RadarSwinLoss, self).__init__()
        self.config = config
        self.range_es = range_es
        self.bbox_size_ori = bbox_size_ori
        self.velocity = velocity
        self.cross_entropy = nn.CrossEntropyLoss()
        self.alpha = 2
        self.beta = 4

    def forward(self, pred, target):

        # print(pred.shape)
        # print(target.shape)
        # pred  : B, 360, 8     : hmap, range, orientation [sin, cos], size [w l], velocity [vr vt]
        # target: B, 360, 8 + 1 : hmap, range, orientation [sin, cos], size [w l], velocity [vr vt], mask
        #apply focal loss between pred[:, 0] and target[:, 0]
        # pred = pred.permute(0, 2, 1).view(-1, 8)
        # target = target.permute(0, 2, 1).view(-1, 8)
        pred = pred.view(-1, 8)
        # assert torch.any(torch.isnan(pred)) == False, 'NaN predictions'
        target = target.view(-1, 9)
        # assert torch.any(torch.isnan(target)) == False, 'NaN transformed target'

        mask = target[:, -1]

        sum_mask = torch.sum(mask)
        # if sum_mask == 0:
        #     return torch.tensor(0.0, requires_grad=True).cuda()
        # assert sum_mask > 0

        mask_pred = pred[mask == 1]
        unmask_pred = pred[mask != 1]
        mask_target = target[mask == 1]
        unmask_target = target[mask != 1]

        # assert torch.any(torch.isnan(((1 - mask_pred[:, 0]) ** self.alpha))) == False, 'NaN first part focal'
        # assert torch.any(torch.isnan(torch.log(mask_pred[:, 0] + 1e-4))) == False, 'NaN second part focal'
        focal_mask_loss = torch.sum(((1 - mask_pred[:, 0]) ** self.alpha) * torch.log(mask_pred[:, 0] + 1e-4))
        # assert torch.isnan(focal_mask_loss) == False, 'NaN focal mask loss'
        # print("fm", focal_mask_loss)
        focal_umask_loss = torch.sum(((1 - unmask_target[:, 0]) ** self.beta) * (unmask_pred[:, 0] ** self.alpha) * torch.log(1 - unmask_pred[:, 0] + 1e-4))
        # assert torch.isnan(focal_umask_loss) == False, 'NaN focal unmask loss'
        # print("fu", focal_umask_loss)
        focal_loss = (focal_mask_loss + focal_umask_loss) * -1 / (sum_mask + 1e-4)
        # print("f", focal_loss)

        #l1 norm
        range_loss = torch.sum(torch.abs(mask_pred[:, 1] - mask_target[:, 1])) / (sum_mask + 1e-4)
        # assert torch.isnan(range_loss) == False, 'NaN range loss'
        
        # print("ra", range_loss)
        
        #should be 2,3 but only 2-sin is enoug, 180-degree symmetry
        bbox_ori_loss = torch.sum(torch.abs(mask_pred[:, 2] - mask_target[:, 2])) / (sum_mask + 1e-4)

        bbox_size_loss = torch.sum(torch.abs(mask_pred[:, 4:6] * mask_pred[:, [1, 1]] - mask_target[:, 4:6] * mask_target[:, [1, 1]])) / (sum_mask + 1e-4)


        # velocity_loss = torch.sum(torch.abs(mask_pred[:, 6] - mask_target[:, 6]))
        velocity_R_loss = torch.sum(torch.abs(mask_pred[:, 6] * mask_pred[:, 1] - mask_target[:, 6] * mask_target[:, 1])) / (sum_mask + 1e-4)
        
        # velocity_T_loss = torch.sum(torch.abs(mask_pred[:, 7] * mask_pred[:, 1] - mask_target[:, 7] * mask_target[:, 1])) / (sum_mask + 1e-4)
        # assert torch.isnan(velocity_loss) == False, 'NaN velocity loss'
        # print("ve", velocity_loss)


        # return focal_loss + range_loss * 0.1
        # return focal_loss + 0.1 * range_loss + 0.1 * velocity_R_loss
        return focal_loss + 0.1 * range_loss + 0.1 * velocity_R_loss + 0.1 * bbox_ori_loss + 0.1 * bbox_size_loss
