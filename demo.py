from src.models.mobileFill import MobileFill
import torch
import numpy as np
from PIL import Image
from collections import OrderedDict
import torch.nn.functional as F
from src.utils.util import tensor2cv,cv2tensor
import yaml
import os
import random
import warnings

def postprocess(imgs_t, to_cv=True):
    imgs_t = (imgs_t + 1.0) * 0.5  # scale to 0 ~ 1
    if not to_cv:
        return imgs_t

    return tensor2cv(imgs_t)


def load_model(config_path, model_path, device = "cuda"):
    with open(config_path, "r") as f:
        opts = yaml.load(f, yaml.FullLoader)
    model = MobileFill(**opts)
    net_state_dict = model.state_dict()
    state_dict = torch.load(model_path, map_location='cpu')["ema_G"]
    new_state_dict = {k: v for k, v in state_dict.items() if k in net_state_dict}
    model.load_state_dict(OrderedDict(new_state_dict), strict=False)
    model.eval().requires_grad_(False).to(device)

    return model, state_dict


def set_random_seed(random_seed=666,deterministic=False):
    if random_seed is not None:
        print("Set random seed as {}".format(random_seed))
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        if deterministic:
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            #for faster training
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True




if __name__ == '__main__':
    # seed = 2023
    # set_random_seed(seed)
    config_path = "./configs/ablations/wo_vit.yaml"
    model_path = './checkpoints/G-step=360000_lr=0.0001_ema_loss=0.4681.pth'
    # model_path1 = './checkpoints/G-step=400000_lr=0.0001_ema_loss=0.4677.pth'
    # model_path2 = './checkpoints/G-step=424000_lr=0.0001_ema_loss=0.4677.pth'
    # model_path3 = './checkpoints/G-step=360000_lr=0.0001_ema_loss=0.4681.pth'
    # model_path3 = "/home/codeoops/CV/inpainting_baseline/checkpoints/MobileFill/celeba-hq/wo_large_kernel.pth"

    model_path1 = './debug_0.pth'
    model_path2 = './debug_1.pth'
    model_path3 = "./debug_2.pth"
    #
    path_list = [model_path1, model_path2, model_path3]
    # path_list = [model_path, model_path, model_path]
    #
    # path_list = [model_path]
    # path_list = [model_path3]

    # model = load_model(config_path = config_path, model_path = model_path)

    # model_list = [model]

    model_list = []
    model, state_dict = load_model(config_path=config_path, model_path=model_path)
    model_list.append(model)
    state_dict_list = []
    for i, path in enumerate(path_list):
        state_dict = torch.load(path, map_location='cpu')
        state_dict_list.append(state_dict)
        # torch.save(state_dict, f"./debug_{i}.pth")

    test_img_path = './example/celeba-hq/imgs/00037_im_truth.jpg'
    test_mask_path = './example/celeba-hq/masks/00037_im_mask.png'
    save_dir = "./debug_result"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = 'cuda'

    img = np.array(Image.open(test_img_path))
    mask = np.array(Image.open(test_mask_path))
    Image.fromarray(img).save(save_dir + "/real.png")
    sample_num = 1

    if len(mask.shape) < 3:
        mask = np.expand_dims(mask, axis=-1)

    imgs_t = cv2tensor([img]).to(device)
    masks_t = cv2tensor([mask]).to(device)

    if masks_t.size(1) > 1:
        masks_t = masks_t[:, 2:3, :, :]

    masks_t = masks_t.repeat(sample_num, 1, 1, 1)
    imgs_t = imgs_t.repeat(sample_num, 1, 1, 1)
    masks_t = 1 - masks_t  # 0 for holes
    imgs_t = imgs_t / 0.5 - 1.0

    masked_imgs = imgs_t * masks_t
    input_imgs = torch.cat((masked_imgs, masks_t), dim=1)

    with torch.no_grad():
        for k, model in enumerate(model_list):
            # for j, path in enumerate(path_list):
            #     model_state_dict = model.state_dict()
            #     state_dict = torch.load(model_path, map_location="cuda")["ema_G"]
            #     # new_state_dict = {k: v for k, v in state_dict.items() if k.startswith("generator")}
            #     # model_state_dict.update(OrderedDict(new_state_dict))
            #     model.load_state_dict(state_dict)
            for j, state_dict in enumerate(state_dict_list):
                model.load_state_dict(state_dict)
                out, ws = model(input_imgs)
                merged_img = imgs_t * masks_t + out["img"] * (1 - masks_t)

                for i in range(sample_num):
                    masked_im_np = postprocess(masked_imgs[i:i + 1])
                    out_np = postprocess(merged_img[i:i + 1])
                    Image.fromarray(masked_im_np[0]).save(save_dir + f"/masked.png")
                    Image.fromarray(out_np[0]).save(save_dir + f"/out_{k}_{j}_{i}.png")

