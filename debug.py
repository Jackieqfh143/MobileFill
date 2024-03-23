from src.utils.util import tensor2cv, cv2tensor
from src.utils.im_process import get_transparent_mask
from src.models.mat import MAT
from src.models.mobileFill import MobileFill
import torch.nn.functional as F
import yaml
import os
import glob
import torch
from collections import OrderedDict
from PIL import Image
import numpy as np
from tqdm import tqdm



if __name__ == '__main__':
    img_path = "/home/codeoops/CV/inpainting_baseline/example/places/Places/thick_512/imgs"
    mask_path = "/home/codeoops/CV/inpainting_baseline/example/places/Places/thick_512/masks"
    save_path = "./debug_result_mat"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = "cuda"
    mat_path = "./checkpoints/MAT/place/Places_512_FullData.pkl"
    mobilefill_path = "/home/codeoops/CV/inpainting_baseline/checkpoints/MobileFill/places/G-step=660000_lr=0.0001_ema_loss=0.495.pth"
    config_path = "./configs/mobilefill.yaml"

    with open(config_path, "r") as f:
        opts = yaml.load(f, yaml.FullLoader)

    mat_model = MAT(model_path=mat_path, device="cuda", targetSize=512)
    mobilefill_model = MobileFill(**opts)
    net_state_dict = mobilefill_model.state_dict()
    state_dict = torch.load(mobilefill_path, map_location='cpu')["ema_G"]
    new_state_dict = {k: v for k, v in state_dict.items() if k in net_state_dict}
    mobilefill_model.load_state_dict(OrderedDict(new_state_dict), strict=False)
    mobilefill_model.eval().requires_grad_(False).to(device)


    imgs = sorted(glob.glob(img_path + "/*.jpg") + glob.glob(img_path + "/*.png"))[:10]
    masks = sorted(glob.glob(mask_path + "/*.jpg") + glob.glob(mask_path + "/*.png"))[:10]

    imgs = [np.array(Image.open(im_path)) for im_path in imgs]
    masks = [np.array(Image.open(mask_path)) for mask_path in masks]

    imgs_t = cv2tensor(imgs)
    masks_t = cv2tensor(masks)

    b,c,h,w = imgs_t.size()

    for i in tqdm(range(b)):
        mask = masks_t[i:i+1,0:1,:,:].to(device)
        mask = 1 - mask
        gt = imgs_t[i:i+1].to(device)

        gt_ = (gt / 0.5) - 1

        mat_res, *_ = mat_model.forward(gt_, mask)

        if gt.size(-1) != 256:
            gt = F.interpolate(gt, 256, mode="bilinear")

        if gt_.size(-1) != 256:
            gt_ = F.interpolate(gt_, 256, mode="bilinear")

        if mask.size(-1) != 256:
            mask = F.interpolate(mask, 256, mode="bilinear")

        if mat_res.size(-1) != 256:
            mat_res = F.interpolate(mat_res, 256, mode="bilinear")

        masked_im = gt_ * mask  # 0 for holes
        input_x = torch.cat((masked_im, mask), dim=1)
        pred_img = mobilefill_model(input_x)[0]["img"]

        output = (1 - mask) * pred_img + gt_ * mask

        gt_img_np = gt[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        # mask_np = (1 - mask[0]).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        # mask_np = np.concatenate([mask_np, mask_np, mask_np], axis=-1)
        # masked_img_np, _ = get_transparent_mask(gt_img_np, mask_np)  # the input mask should be 1 for holes
        masked_im = gt * mask + (1 - mask)
        masked_img_np = tensor2cv(masked_im)[0]
        Image.fromarray(masked_img_np).save(save_path + f'/{i:0>5d}_im_masked.jpg')
        Image.fromarray(gt_img_np).save(save_path + f'/{i:0>5d}_im_truth.jpg')

        mat_res = (mat_res + 1.0) * 0.5
        output = (output + 1.0) * 0.5
        mat_res_np = tensor2cv(mat_res)[0]
        output_np = tensor2cv(output)[0]
        Image.fromarray(mat_res_np).save(save_path + f'/{i:0>5d}_mat_im_out.jpg')
        Image.fromarray(output_np).save(save_path + f'/{i:0>5d}_mobilefill_im_out.jpg')











