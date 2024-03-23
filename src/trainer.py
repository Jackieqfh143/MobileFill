import os
import shutil
import torch
import torch.nn as nn
import numpy as np
from src.models.baseModel import BaseModel
from src.evaluate.loss import l1_loss, Dis_loss_mask,Gen_loss,Dis_loss, style_seeking_loss, ori_mode_seeking_loss
from src.utils.util import checkDir,tensor2cv
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from PIL import Image
from torch_ema import ExponentialMovingAverage
from pytorch_wavelets import DWTInverse, DWTForward
from src.models.stylegan import StyleGANModel
from src.utils.util import find_model_using_name
from src.utils.diffaug import rand_cutout
import kornia
import yaml


class MyTrainer(BaseModel):
    def __init__(self,opt):
        super(MyTrainer, self).__init__(opt)
        self.count = 0
        self.opt = opt
        self.lossNet = find_model_using_name("src.evaluate.pcp", model_name=opt.pl_loss_type,
                                             weights_path=opt.lossNetDir) #network for calculate percptual loss
        self.flops = None
        self.device = self.accelerator.device
        self.lossNet = self.lossNet.to(self.device)
        self.recorder = SummaryWriter(self.log_path)
        self.current_lr = opt.lr
        self.current_d_lr = opt.d_lr

        with open(self.opt.config_path, "r") as f:
            opts = yaml.load(f, yaml.FullLoader)

        self.G_Net = find_model_using_name("src.models.mobileFill", model_name=opt.Generator,
                                           **opts)
        self.D_Net = find_model_using_name("src.models.discriminator", model_name=opt.Discriminator, size = 256)

        self.G_opt = torch.optim.AdamW(self.G_Net.parameters(), opt.lr,
                                       betas=(opt.beta_g_min, opt.beta_g_max))
        self.D_opt = torch.optim.AdamW(self.D_Net.parameters(), lr=opt.d_lr,
                                       betas=(opt.beta_d_min, opt.beta_d_max))
        if opt.restore_training:
            self.load()

        self.ema_G = ExponentialMovingAverage(self.G_Net.parameters(), decay=0.995)
        self.ema_G.to(self.device)

        self.acc_args = [self.G_Net, self.D_Net, self.G_opt, self.D_opt]

        if opt.enable_gan_aug:
            self.gan_teacher = StyleGANModel(model_path=opt.gan_teacher_path, device=self.device,
                                             targetSize=self.opt.targetSize)

        if opt.enable_teacher:
            self.teacher = find_model_using_name("src.models.mat", model_name=opt.Teacher,
                                                 model_path=opt.teacher_path, device=self.device,
                                                 targetSize=self.opt.teacher_target_size)

        self.img_augment = nn.Sequential(kornia.augmentation.RandomHorizontalFlip(),
                                        kornia.augmentation.RandomHue(p=0.5),
                                        kornia.augmentation.RandomSaturation(p=0.5)
                                        )

        self.dwt = DWTForward(J=1, mode='zero', wave='db1').to(self.device)
        self.idwt = DWTInverse(mode="zero", wave="db1").to(self.device)

        self.lossDict = {}
        self.print_loss_dict = {}
        self.im_dict = {}
        self.val_im_dict = {}
        self.isTraining = False

    def train(self):
        self.G_Net.train()
        self.D_Net.train()
        self.isTraining = True

    def eval(self):
        self.G_Net.eval()
        self.isTraining = False

    def set_input(self,imgs, masks):
        masks = masks[:, 2:3, :, :]
        real_imgs = self.preprocess(imgs)  # scale to -1 ~ 1

        # if self.isTraining and self.opt.teacher_target_size == 512:
        #     imgs_512 = self.preprocess(imgs_512)
        #     masks_512 = masks_512[:, 2:3, :, :]

        if self.opt.enable_gan_aug:
            if self.isTraining and np.random.binomial(1, 0.5) > 0:
                z = torch.randn(real_imgs.size(0), 512).to(self.device)
                real_imgs = self.img_augment(self.make_sample(z))

        real_imgs_ = F.interpolate(real_imgs, self.G_Net.target_size, mode="bilinear")
        masks_ = F.interpolate(masks, self.G_Net.target_size, mode="nearest")

        masked_im = real_imgs_ * masks_  # 0 for holes
        input_imgs = torch.cat((masked_im, masks_),dim=1)

        return input_imgs, real_imgs, masks

    @torch.no_grad()
    def make_sample(self, z):
        samples, _ = self.gan_teacher.sample(z)
        return samples

    @torch.no_grad()
    def make_example(self,imgs,masks):
        samples, all_imgs, stg1s = self.teacher.forward(imgs,masks)
        return samples, all_imgs, stg1s

    # @torch.no_grad()
    # def make_rand_example(self, imgs, ratio = 0.5):
    #     masks = rand_cutout(imgs, ratio)
    #     samples = self.teacher.forward(imgs, masks)
    #     return samples

    def get_all_imgs(self, out):
        out_freqs = out["freq"]
        all_imgs = []
        for freq in out_freqs:
            img = F.tanh(self.G_Net.generator.dwt_to_img(freq))
            all_imgs.append(img)

        return all_imgs

    def forward(self,batch,count):
        self.count = count
        input_imgs, real_imgs, masks = self.set_input(*batch)
        out2, w2 = self.G_Net(input_imgs)
        out1,w1 = self.G_Net(input_imgs)

        real_imgs1 = real_imgs
        real_imgs2 = real_imgs
        if self.isTraining and self.opt.enable_teacher:
            # teach_imgs = imgs_512 if self.opt.teacher_target_size == 512 else real_imgs
            # teach_masks = masks_512 if self.opt.teacher_target_size == 512 else masks
            real_imgs1, _, _ = self.make_example(real_imgs, masks)
            real_imgs2, _, _ = self.make_example(real_imgs, masks)


        fake_imgs1 = out1["img"]
        fake_freq1 = out1["freq"][-1]

        if real_imgs1.size(-1) != fake_imgs1.size(-1):
            real_imgs1 = F.interpolate(real_imgs1, fake_imgs1.size(-1), mode="bilinear")

        if masks.size(-1) != fake_imgs1.size(-1):
            masks = F.interpolate(masks, fake_imgs1.size(-1), mode="nearest")

        if real_imgs2.size(-1) != fake_imgs1.size(-1):
            real_imgs2 = F.interpolate(real_imgs2, fake_imgs1.size(-1), mode="bilinear")


        comp_imgs1 = real_imgs1 * masks + fake_imgs1 * (1 - masks)
        fake_imgs2 = out2["img"]
        fake_freq2 = out2["freq"][-1]
        comp_imgs2 = real_imgs2 * masks + fake_imgs2 * (1 - masks)

        return masks, real_imgs1,real_imgs2,comp_imgs1,comp_imgs2,fake_freq1,fake_freq2,w1,w2

    def backward_G(self, real_imgs, fake_imgs, fake_freq, masks):
        # if self.opt.use_rec_freq_loss:
        #     real_freq = self.img_to_dwt(real_imgs)

        # if real_imgs.size(-1) != 256:
        #     real_imgs = F.interpolate(real_imgs, 256, mode="bilinear")
        #
        # if fake_imgs.size(-1) != 256:
        #     fake_imgs = F.interpolate(fake_imgs, 256, mode="bilinear")
        #
        # if masks.size(-1) != 256:
        #     masks = F.interpolate(masks, 256, mode="nearest")

        g_loss_list = []
        g_loss_name_list = []
        pred_feats = None
        gt_feats = None
        if self.opt.use_rec_loss:
            if not self.opt.enable_teacher:
                rec_loss = self.opt.lambda_valid * l1_loss(real_imgs, fake_imgs,
                                                                masks)  # keep background unchanged
            else:
                rec_loss = self.opt.lambda_valid * l1_loss(real_imgs,fake_imgs)

            g_loss_list.append(rec_loss)
            g_loss_name_list.append("rec_loss")

        if self.opt.use_rec_freq_loss:
            if not self.opt.enable_teacher:
                # frequentcy reconstruct loss
                real_freq = self.img_to_dwt(real_imgs)
                mask_ = F.interpolate(masks, size=real_freq.size(-1), mode="nearest")
                rec_freq_loss = self.opt.lambda_hole * l1_loss(real_freq, fake_freq, mask_)
            else:
                # frequentcy reconstruct loss
                real_freq = self.img_to_dwt(real_imgs)
                rec_freq_loss = self.opt.lambda_hole * l1_loss(real_freq, fake_freq)
            g_loss_list.append(rec_freq_loss)
            g_loss_name_list.append("rec_freq_loss")

        if self.opt.use_gan_loss:
            dis_fake, fake_d_feats = self.D_Net(fake_imgs, masks)
            gen_loss = Gen_loss(dis_fake, type=self.opt.gan_loss_type)
            gen_loss = self.opt.lambda_gen * gen_loss
            g_loss_list.append(gen_loss)
            g_loss_name_list.append("gen_loss")

        if self.opt.use_perc_loss:
            perc_loss, pred_feats, gt_feats = self.lossNet(fake_imgs, real_imgs)
            perc_loss = self.opt.lambda_perc * perc_loss
            g_loss_list.append(perc_loss)
            g_loss_name_list.append("perc_loss")

        # if self.opt.use_multi_perc_loss:
        #     multi_perc_loss = 0.0
        #     for fake_im in fake_all_imgs:
        #         for real_im in real_all_imgs:
        #             if fake_im.size(-1) == real_im.size(-1):
        #                 perc_loss, *_ = self.lossNet(fake_imgs, real_imgs)
        #                 multi_perc_loss += perc_loss
        #
        #     multi_perc_loss = self.opt.lambda_perc * multi_perc_loss
        #     g_loss_list.append(multi_perc_loss)
        #     g_loss_name_list.append("multi_perc_loss")

        G_loss = 0.0
        for loss_name,loss in zip(g_loss_name_list,g_loss_list):
            G_loss += loss

        return G_loss, g_loss_name_list, g_loss_list, pred_feats, gt_feats

    def backward_D(self,real_imgs, fake_imgs, masks):
        # if real_imgs.size(-1) != 256:
        #     real_imgs = F.interpolate(real_imgs, 256, mode="bilinear")
        #
        # if fake_imgs.size(-1) != 256:
        #     fake_imgs = F.interpolate(fake_imgs, 256, mode="bilinear")
        #
        # if masks.size(-1) != 256:
        #     masks = F.interpolate(masks, 256, mode="nearest")

        if self.opt.gan_loss_type == 'R1':
            real_imgs.requires_grad = True
            masks.requires_grad = True

        dis_real, self.real_d_feats = self.D_Net(real_imgs, masks)

        if self.opt.D_input_type == "comp_img" and "StyleGAN" not in self.opt.Discriminator:
            dis_comp, _ = self.D_Net(fake_imgs.detach(), masks.detach())

            dis_loss, r1_loss = Dis_loss_mask(dis_real, dis_comp, (1 - masks), real_bt=real_imgs,
                                              type=self.opt.gan_loss_type, lambda_r1=self.opt.lambda_r1)
        else:
            dis_fake, _ = self.D_Net(fake_imgs.detach(), masks.detach())
            dis_loss, r1_loss = Dis_loss(dis_real, dis_fake, real_bt=real_imgs,
                                         type=self.opt.gan_loss_type, lambda_r1=self.opt.lambda_r1)

        return dis_loss, r1_loss

    def optimize_params(self,batch,count):
        masks, real_imgs1,real_imgs2,comp_imgs1,comp_imgs2,fake_freq1,fake_freq2,w1,w2 = self.forward(batch,count)

        if self.opt.use_gan_loss:
            with self.accelerator.accumulate(self.D_Net):
                self.D_opt.zero_grad()
                dis_loss1, r1_loss1 = self.backward_D(real_imgs1,comp_imgs1,masks)
                dis_loss2, r1_loss2 = self.backward_D(real_imgs2, comp_imgs2,masks)
                dis_loss = dis_loss1 + dis_loss2
                r1_loss = r1_loss1 + r1_loss2

                self.lossDict['dis_loss'] = dis_loss.item()
                self.lossDict['r1_loss'] = r1_loss.item()
                self.accelerator.backward(dis_loss)
                self.D_opt.step()

        with self.accelerator.accumulate(self.G_Net):
            self.G_opt.zero_grad()
            G_loss1, g_loss_name_list1, g_loss_list1, pred_feats1, gt_feats1 = self.backward_G(real_imgs1, comp_imgs1, fake_freq1,masks
                                                                                               )
            G_loss2, g_loss_name_list2, g_loss_list2, pred_feats2, gt_feats2 = self.backward_G(real_imgs2, comp_imgs2, fake_freq2,masks
                                                                                               )
            mode_seeking_loss = self.opt.pcp_penalty * style_seeking_loss(gt_feats1, gt_feats2,
                                                                                       pred_feats1,
                                                                                       pred_feats2,
                                                                                       w1, w2)
            #
            # mode_seeking_loss = self.opt.pcp_penalty * ori_mode_seeking_loss(comp_imgs1, comp_imgs2, w1, w2)

            G_loss = G_loss1 + G_loss2 + mode_seeking_loss
            self.lossDict['mode_seek'] = mode_seeking_loss.item()
            for i in range(len(g_loss_name_list1)):
                loss_name = g_loss_name_list2[i]
                self.lossDict[loss_name] = g_loss_list1[i].item() + g_loss_list2[i].item()

            self.accelerator.backward(G_loss)
            self.G_opt.step()
            # if self.opt.enable_ema:
            self.ema_G.update(self.G_Net.parameters())

        self.logging()

    def adjust_learning_rate(self, lr_in, min_lr, optimizer, epoch, lr_factor=0.95, warm_up=False, name='lr'):
        if not warm_up:
            lr = max(lr_in * lr_factor, float(min_lr))
        else:
            lr = max(lr_in * (epoch / int(self.opt.warm_up_epoch)), float(min_lr))

        print(f'Adjust learning rate to {lr:.5f}')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        setattr(self, f'current_{name}', lr)

    @torch.no_grad()
    def validate(self,batch,count):
        self.val_count = count
        input_imgs, real_imgs, masks, *_ = self.set_input(*batch)

        real_imgs = F.interpolate(real_imgs, self.G_Net.target_size, mode="bilinear")
        masks = F.interpolate(masks, self.G_Net.target_size, mode="nearest")
        unwrap_model = self.accelerator.unwrap_model(self.G_Net)
        unwrap_model = unwrap_model.to(self.device)
        out = unwrap_model(input_imgs)[0]['img']
        fake_imgs = real_imgs * masks + out * (1 - masks)
        fake_imgs = self.postprocess(fake_imgs)
        real_imgs = self.postprocess(real_imgs)
        masked_imgs = real_imgs * masks

        if self.opt.record_val_imgs:
            self.val_im_dict['fake_imgs'] = fake_imgs.cpu().detach()
            self.val_im_dict['real_imgs'] = real_imgs.cpu().detach()
            self.val_im_dict['masked_imgs'] = masked_imgs.cpu().detach()

        self.logging()

        return tensor2cv(real_imgs),tensor2cv(fake_imgs),tensor2cv(masked_imgs)

    # def get_current_imgs(self):
    #     self.im_dict['real_imgs'] = self.real_imgs.cpu().detach()
    #     self.im_dict['masked_imgs'] = (self.real_imgs * self.mask).cpu().detach()

    def logging(self):
        for lossName, lossValue in self.lossDict.items():
            self.recorder.add_scalar(lossName, lossValue, self.count)

        if self.print_loss_dict == {}:
            temp = {k:[] for k in self.lossDict.keys()}
            self.print_loss_dict.update(temp)
            self.print_loss_dict['r1_loss'] = []
        else:
            for k,v in self.lossDict.items():
                if k in self.print_loss_dict.keys():
                    self.print_loss_dict[k].append(v)


        # if self.opt.record_training_imgs:
        #     if self.count % self.opt.save_im_step == 0:
        #         self.get_current_imgs()
        #         for im_name,im in self.im_dict.items():
        #             im_grid = vutils.make_grid(im, normalize=False, scale_each=True)
        #             self.recorder.add_image(im_name,im_grid,self.count)
        #
        # if self.opt.record_val_imgs:
        #     if self.count % self.opt.val_step == 0:
        #         for im_name, im in self.val_im_dict.items():
        #             im_grid = vutils.make_grid(im, normalize=False, scale_each=True)
        #             self.recorder.add_image(im_name, im_grid, self.val_count)

    def reduce_loss(self):
        for k, v in self.print_loss_dict.items():
            if len(v) != 0:
                self.print_loss_dict[k] = sum(v) / len(v)
            else:
                self.print_loss_dict[k] = 0.0


    #save validate imgs
    def save_results(self,val_real_ims,val_fake_ims,val_masked_ims=None):
        im_index = 0
        val_save_dir = os.path.join(self.val_saveDir, 'val_results')
        if os.path.exists((val_save_dir)):
            shutil.rmtree(val_save_dir)
        checkDir([val_save_dir])
        for real_im, comp_im, masked_im in zip(val_real_ims, val_fake_ims, val_masked_ims):
            Image.fromarray(real_im).save(val_save_dir + '/{:0>5d}_im_truth.jpg'.format(im_index))
            Image.fromarray(comp_im).save(val_save_dir + '/{:0>5d}_im_out.jpg'.format(im_index))
            Image.fromarray(masked_im).save(val_save_dir + '/{:0>5d}_im_masked.jpg'.format(im_index))
            im_index += 1

    def load(self):
        with self.accelerator.main_process_first():
            if os.path.isdir(self.saveDir):
                if self.opt.load_from_iter != None:
                    model_path = self.find_model_by_iter(self.saveDir, self.opt.load_from_iter)
                else:
                    if not self.opt.load_last:
                        model_path = self.find_model(self.saveDir, model_name='best')
                    else:
                        model_path = self.find_model(self.saveDir, model_name='last')
            else:
                model_path = self.saveDir

            self.accelerator.print('load from checkpoints: ', model_path)
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.G_Net.load_state_dict(state_dict["g"], strict=True)
                # self.ema_G.load_state_dict(state_dict["ema_G"])
                self.D_Net.load_state_dict(state_dict["d"], strict=True)
                self.G_opt.load_state_dict(state_dict["g_optim"])
                self.D_opt.load_state_dict(state_dict["d_optim"])
            except Exception as e:
                print('Failed to load from checkpoints: ', model_path)
                raise e

    # save checkpoint
    def save_network(self, loss_mean_val, val_type='default'):
        save_path = os.path.join(self.saveDir,
                                 "G-step={}_lr={}_{}_loss={}.pth".format(self.count + 1, round(self.current_lr, 6),
                                                                         val_type, loss_mean_val))
        self.accelerator.print('saving network...')
        g_net = self.accelerator.unwrap_model(self.G_Net)
        d_net = self.accelerator.unwrap_model(self.D_Net)
        ema_g = self.accelerator.unwrap_model(self.ema_G)
        g_opt = self.accelerator.unwrap_model(self.G_opt)
        d_opt = self.accelerator.unwrap_model(self.D_opt)

        ema_g.store(g_net.parameters())
        ema_g.copy_to(g_net.parameters())
        ema_state_dict = g_net.state_dict()
        ema_g.restore(g_net.parameters())
        self.accelerator.save({
            "g": g_net.state_dict(),
            "d": d_net.state_dict(),
            "ema_G": ema_state_dict,
            "g_optim": g_opt.state_dict(),
            "d_optim": d_opt.state_dict(),
        }, save_path)
        self.accelerator.print('saving network done. ')

    def normalize(self,t, range=(-1, 1)):
        t.clamp_(min=range[0], max=range[1])
        return t

    def preprocess(self, x):
        return x / 0.5 - 1.0

    def postprocess(self, x):
        return (x + 1.0) * 0.5

    def requires_grad(self,model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def img_to_dwt(self, img):
        low, high = self.dwt(img)
        b, _, _, h, w = high[0].size()
        high = high[0].view(b, -1, h, w)
        freq = torch.cat([low, high], dim=1)
        return freq

    def dwt_to_img(self, img):
        b, c, h, w = img.size()
        low = img[:, :3, :, :]
        high = img[:, 3:, :, :].view(b, 3, 3, h, w)
        return self.idwt((low, [high]))









