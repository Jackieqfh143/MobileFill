from torch import nn
import torch
import torchvision.models as models
import torch.nn.functional as F

def l1_loss(f1, f2, mask=1):
    return torch.mean(torch.abs(f1 - f2) * mask)

#Non-saturate R1 GP (penalize only real data)
def make_r1_gp(discr_real_pred, real_batch):
    real_batch.requires_grad = True
    if torch.is_grad_enabled():
        grad_real = torch.autograd.grad(outputs=discr_real_pred.sum(), inputs=real_batch, create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.shape[0], -1).norm(2, dim=1) ** 2).mean()
    else:
        grad_penalty = 0

    return grad_penalty

def Dis_loss(pos, neg,type = 'Softplus',real_bt=None,lambda_r1=0.001):
    if type == 'Hinge':
        dis_loss = torch.mean(F.relu(1. - pos)) + torch.mean(F.relu(1. + neg))
    elif type == 'Softplus':
        dis_loss = F.softplus(- pos).mean() + F.softplus(neg).mean()
    elif type == 'R1':
        grad_penalty = make_r1_gp(pos, real_bt) *lambda_r1
        dis_loss = (F.softplus(- pos) + F.softplus(neg) + grad_penalty).mean()
    elif type == 'MSE':
        real_target = torch.zeros_like(pos)
        fake_target = torch.ones_like(neg)
        dis_loss = F.mse_loss(pos, real_target).mean() + F.mse_loss(neg, fake_target).mean()
    else:
        #BCE loss
        real_target = torch.zeros_like(pos)
        fake_target = torch.ones_like(neg)
        dis_loss = F.binary_cross_entropy(pos, real_target).mean() + F.binary_cross_entropy(neg, fake_target).mean()


    if type == 'R1':
        return dis_loss,grad_penalty.mean()
    else:
        return dis_loss

def Dis_loss_mask(pos,neg,mask,type = 'Softplus',real_bt=None,lambda_r1=0.001):
    if neg.shape[3] != mask.shape[3]:
        mask = F.interpolate(mask, size=neg.shape[2:], mode='nearest')

    #input mask (1 for fake part )
    if type == 'Hinge':
        dis_loss = F.relu(1. - pos) + mask * F.relu(1 + neg) + (1 - mask) * F.relu(1. - neg)
    elif type == 'Softplus':
        dis_loss = F.softplus(- pos) + F.softplus(neg) * mask + (1 - mask) * F.softplus(-neg)
    elif type == 'R1':
        grad_penalty = make_r1_gp(pos, real_bt) * lambda_r1
        dis_loss = F.softplus(- pos) + F.softplus(neg) * mask + (1 - mask) * F.softplus(-neg) + grad_penalty
    elif type == 'MSE':
        real_target = torch.zeros_like(pos)
        dis_loss = F.mse_loss(pos,real_target) + F.mse_loss(neg,mask)
    else:
        #BCE loss
        real_target = torch.zeros_like(pos)
        dis_loss = F.binary_cross_entropy(pos, real_target) + F.binary_cross_entropy(neg, mask)

    if type == 'R1':
        return dis_loss.mean(), grad_penalty.mean()
    else:
        return dis_loss.mean()

def Gen_loss(neg,type = 'Softplus'):
    if type == 'Hinge':
        gen_loss = -torch.mean(neg)
    elif type == 'Softplus' or type =='R1':
        gen_loss = F.softplus(-neg).mean()  #softplus is the smooth version of Relu()
    elif type == 'MSE':
        target = torch.zeros_like(neg)
        # MSE loss
        gen_loss = F.mse_loss(neg, target).mean()
    else:
        #BCE loss
        target = torch.zeros_like(neg)
        # BCE loss
        gen_loss = F.binary_cross_entropy(neg, target).mean()

    return gen_loss

def Gen_loss_mask(neg,mask,type = 'Softplus'):

    if neg.shape[3] != mask.shape[3]:
        mask = F.interpolate(mask,size=neg.shape[2:],mode='nearest')

    # input mask (1 for fake part )
    if type == 'Hinge':
        gen_loss = -neg * mask
    elif type == 'Softplus' or type =='R1':
        gen_loss = F.softplus(-neg)
        gen_loss = gen_loss * mask
    elif type == 'MSE':
        target = torch.zeros_like(neg)
        # MSE loss
        gen_loss = F.mse_loss(neg, target)
        gen_loss = gen_loss * mask
    else:
        target = torch.zeros_like(neg)
        #BCE loss
        gen_loss = F.binary_cross_entropy(neg,target)
        gen_loss = gen_loss * mask
    return gen_loss.mean()

def style_seeking_loss(gt_feats1, gt_feats2,
                         pred_feats1, pred_feats2, ws1, ws2):
        eps = 1 * 1e-5
        # pcp_loss_gt range: 0.05 ~ 0.5
        pcp_loss_gt = torch.stack([F.mse_loss(cur_pred, cur_target)
                                   for cur_pred, cur_target
                                   in zip(gt_feats1, gt_feats2)]).sum()

        pcp_loss_fake = torch.stack([F.mse_loss(cur_pred, cur_target)
                                     for cur_pred, cur_target
                                     in zip(pred_feats1, pred_feats2)]).sum()

        # ws_loss = torch.mean(torch.abs(ws1 - ws2))
        # loss = pcp_loss / ws_sim
        # loss_lz = pcp_loss_gt * (1 / (pcp_loss_fake + eps) + 1 / (ws_loss + eps))

        loss_lz = pcp_loss_gt * (1 / (pcp_loss_fake + eps))
        return loss_lz

def ori_mode_seeking_loss(img1, img2, ws1, ws2):
    eps = 1 * 1e-5
    mae_loss = torch.mean(torch.abs(img1 - img2))
    ws_loss = torch.mean(torch.abs(ws1 - ws2))
    loss = mae_loss / ws_loss
    loss_lz = 1 / (loss + eps)
    return loss_lz