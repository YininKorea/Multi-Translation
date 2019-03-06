"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, VAEGen,Ins_Discriminator
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import numpy as np
import cv2
save_name = "./Data/train_a_identity_2.npy"
identity_a = np.load(save_name)
# print(save_name)
save_name = "./Data/train_b_identity_2.npy"
identity_b = np.load(save_name)






# print(save_name)
original_path = "./Data/result_outputs/train/"
isExist = os.path.exists(original_path)
if not isExist:
    os.mkdir(original_path)
original_path = "./Data/result_outputs/test/"
isExist = os.path.exists(original_path)
if not isExist:
    os.mkdir(original_path)
device = torch.device('cuda:3')
class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.gen_ins_a=AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])
        self.gen_ins_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])

        self.dis_ins_a=Ins_Discriminator(hyperparameters['Ins_Dis'],3)
        self.dis_ins_b = Ins_Discriminator(hyperparameters['Ins_Dis'], 3)
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        self.style_dim = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        self.s_ins_a=torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_ins_b=torch.randn(display_size, self.style_dim, 1, 1).cuda()


        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']

        com_dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        com_gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        ins_dis_params = list(self.dis_ins_a.parameters()) + list(self.dis_ins_b.parameters())
        ins_gen_params = list(self.gen_ins_a.parameters()) + list(self.gen_ins_b.parameters())

        self.com_dis_opt = torch.optim.Adam([p for p in com_dis_params if p.requires_grad],
                                            lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.com_gen_opt = torch.optim.Adam([p for p in com_gen_params if p.requires_grad],
                                            lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.ins_dis_opt = torch.optim.Adam([p for p in ins_dis_params if p.requires_grad],
                                            lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.ins_gen_opt = torch.optim.Adam([p for p in ins_gen_params if p.requires_grad],
                                            lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.com_dis_scheduler = get_scheduler(self.com_dis_opt, hyperparameters)
        self.com_gen_scheduler = get_scheduler(self.com_gen_opt, hyperparameters)
        self.ins_dis_scheduler = get_scheduler(self.ins_dis_opt, hyperparameters)
        self.ins_gen_scheduler = get_scheduler(self.ins_gen_opt, hyperparameters)


        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        c_a, s_a_fake = self.gen_a.encode(x_a)
        c_b, s_b_fake = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        self.train()
        return x_ab, x_ba

    def Gen_update(self,x1,x2,y1,y2,cfg):
        #print("it's time for common generative network updating!")
        self.com_gen_update(x1,y1,cfg)
        #print("it's time for Ins generative network updating!")
        self.Ins_gen_update(x2,y2,cfg)

    def Dis_update(self, x1, x2, y1,y2, cfg):
        #print("it's time for common dis network updating!")
        self.com_dis_update(x1, y1, cfg)
        #print("it's time for ins dis network updating!")
        #print(x2.shape)
        #print(y2.shape)
        self.ins_dis_update(x2,y2, cfg)

    def Ins_gen_update(self,Ins_a,Ins_b,hyperparameters):
        #print("start ins gen update")
        self.ins_gen_opt.zero_grad()
        s_ins_a = Variable(torch.randn(Ins_a.size(0), self.style_dim, 1, 1).cuda())
        s_ins_b = Variable(torch.randn(Ins_b.size(0), self.style_dim, 1, 1).cuda())

        c_ins_a, s_ins_a_prime=self.gen_ins_a.encode(Ins_a)
        Ins_a_recon=self.gen_ins_a.decode(c_ins_a, s_ins_a_prime)

        c_ins_b, s_ins_b_prime=self.gen_ins_a.encode(Ins_a)
        Ins_b_recon=self.gen_ins_a.decode(c_ins_b, s_ins_b_prime)

        ins_ab=self.gen_ins_b.decode(c_ins_a,s_ins_b)
        ins_ba=self.gen_ins_a.decode(c_ins_b,s_ins_a)

        # encode again
        c_ins_b_recon, s_ins_a_recon = self.gen_a.encode(ins_ba)
        c_ins_a_recon, s_ins_b_recon = self.gen_b.encode(ins_ab)
        # decode again (if needed)

        ins_aba = self.gen_a.decode(c_ins_a_recon, s_ins_a_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None
        ins_bab = self.gen_b.decode(c_ins_b_recon, s_ins_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

        self.loss_gen_recon_ins_a = self.recon_criterion(Ins_a_recon, Ins_a)
        self.loss_gen_recon_ins_b = self.recon_criterion(Ins_b_recon, Ins_b)
        self.loss_gen_recon_s_ins_a = self.recon_criterion(s_ins_a_recon, s_ins_a)
        self.loss_gen_recon_s_ins_b = self.recon_criterion(s_ins_b_recon, s_ins_b)
        self.loss_gen_recon_c_ins_a = self.recon_criterion(c_ins_a_recon, c_ins_a)
        self.loss_gen_recon_c_ins_b = self.recon_criterion(c_ins_b_recon, c_ins_b)
        self.loss_gen_cycrecon_ins_a = self.recon_criterion(ins_aba, Ins_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_ins_b = self.recon_criterion(ins_bab, Ins_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_ins_a = self.dis_a.calc_gen_loss(ins_ba)
        self.loss_gen_adv_ins_b = self.dis_b.calc_gen_loss(ins_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_ins_a = self.compute_vgg_loss(self.vgg, ins_ba, ins_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_ins_b = self.compute_vgg_loss(self.vgg, ins_ab, ins_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_ins_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_ins_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_ins_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_ins_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_ins_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_ins_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_ins_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_ins_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_ins_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_ins_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_ins_b + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_ins_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_ins_b

        self.loss_ins_gen_total.backward()
        self.ins_gen_opt.step()

    def com_gen_update(self, x_a, x_b, hyperparameters):
        self.com_gen_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(c_a_recon, s_a_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(c_b_recon, s_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b
        self.loss_gen_total.backward()
        self.com_gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b,ins_a,ins_b,mode):
        self.eval()

        #print(mode)
        save_name = "./Data/" + str(mode) + "_a_identity_2.npy"
        identity_a = np.load(save_name)
        #print(save_name)
        save_name = "./Data/" + str(mode) + "_b_identity_2.npy"
        identity_b = np.load(save_name)
        #print(save_name)
        original_path = "./Data/result_outputs/" + mode + ''
        isExist = os.path.exists(original_path)
        if not isExist:
            os.mkdir(original_path)

        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_ins_a1 = Variable(self.s_ins_a)
        s_ins_b1 = Variable(self.s_ins_b)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        s_ins_a2 = Variable(torch.randn(ins_a.size(0), self.style_dim, 1, 1).cuda())
        s_ins_b2 = Variable(torch.randn(ins_b.size(0), self.style_dim, 1, 1).cuda())

        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        ins_a_recon, ins_b_recon, ins_ba1, ins_ba2, ins_ab1, ins_ab2 = [], [], [], [], [], []
        ori_a, ori_b, recon_a, recon_b, \
        trans_com_ab1_ins_ab1, trans_com_ab1_ins_ab2,trans_com_ab2_ins_ab1,trans_com_ab2_ins_ab2,\
        trans_com_ba1_ins_ba1, trans_com_ba1_ins_ba2,trans_com_ba2_ins_ba1,trans_com_ba2_ins_ba2,= [], [], [], [], [], [],[], [], [], [],[], []

        for i in range(x_a.size(0)):

            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            c_ins_a, s_ins_a_fake = self.gen_ins_a.encode(ins_a[i].unsqueeze(0))
            c_ins_b, s_ins_b_fake = self.gen_ins_b.encode(ins_b[i].unsqueeze(0))

            sample_com_a_recon=self.gen_a.decode(c_a, s_a_fake)
            sample_com_b_recon = self.gen_b.decode(c_b, s_b_fake)
            sample_trans_com_ba1_recon = self.gen_a.decode(c_b, s_a1[i].unsqueeze(0))
            sample_trans_com_ba2_recon = self.gen_a.decode(c_b, s_a2[i].unsqueeze(0))
            sample_trans_com_ab1_recon = self.gen_b.decode(c_a, s_b1[i].unsqueeze(0))
            sample_trans_com_ab2_recon = self.gen_b.decode(c_a, s_b2[i].unsqueeze(0))

            sample_ins_a_recon = self.gen_ins_a.decode(c_ins_a,s_ins_a_fake)
            sample_ins_b_recon = self.gen_ins_b.decode(c_ins_b,s_ins_b_fake)

            sample_trans_ins_ba1_recon = self.gen_ins_a.decode(c_ins_b, s_ins_a1[i].unsqueeze(0))
            sample_trans_ins_ba2_recon = self.gen_ins_a.decode(c_ins_b, s_ins_a2[i].unsqueeze(0))
            sample_trans_ins_ab1_recon = self.gen_ins_b.decode(c_ins_a, s_ins_b1[i].unsqueeze(0))
            sample_trans_ins_ab2_recon = self.gen_ins_b.decode(c_ins_a, s_ins_b2[i].unsqueeze(0))



            x_a_recon.append(sample_com_a_recon)
            x_b_recon.append(sample_com_b_recon)
            x_ba1.append(sample_trans_com_ba1_recon)
            x_ba2.append(sample_trans_com_ba2_recon)
            x_ab1.append(sample_trans_com_ab1_recon)
            x_ab2.append(sample_trans_com_ab2_recon)


            ins_a_recon.append(sample_ins_a_recon)
            ins_b_recon.append(sample_ins_b_recon)
            ins_ba1.append(sample_trans_ins_ba1_recon)
            ins_ba2.append(sample_trans_ins_ba2_recon)
            ins_ab1.append(sample_trans_ins_ab1_recon)
            ins_ab2.append(sample_trans_ins_ab2_recon)

            ori_a.append(self.ori_pad(ins_a[i],x_a[i],identity_a[i]))
            ori_b.append(self.ori_pad(ins_b[i],x_b[i], identity_b[i]))
            #print(sample_ins_a_recon.shape)
            #print("sample")
            recon_a.append(self.pad(sample_ins_a_recon, sample_com_a_recon, identity_a[i]))
            recon_b.append(self.pad(sample_ins_b_recon, sample_com_b_recon, identity_b[i]))

            trans_com_ab1_ins_ab1.append(self.pad(sample_trans_ins_ab1_recon, sample_trans_com_ab1_recon, identity_b[i]))
            trans_com_ab1_ins_ab2.append(self.pad(sample_trans_ins_ab2_recon, sample_trans_com_ab1_recon, identity_b[i]))
            trans_com_ab2_ins_ab1.append(self.pad(sample_trans_ins_ab1_recon, sample_trans_com_ab2_recon, identity_b[i]))
            trans_com_ab2_ins_ab2.append(self.pad(sample_trans_ins_ab2_recon, sample_trans_com_ab2_recon, identity_b[i]))

            trans_com_ba1_ins_ba1.append(self.pad(sample_trans_ins_ba1_recon, sample_trans_com_ba1_recon, identity_a[i]))
            trans_com_ba1_ins_ba2.append(self.pad(sample_trans_ins_ba2_recon, sample_trans_com_ba1_recon, identity_a[i]))
            trans_com_ba2_ins_ba1.append(self.pad(sample_trans_ins_ba1_recon, sample_trans_com_ba2_recon, identity_a[i]))
            trans_com_ba2_ins_ba2.append(self.pad(sample_trans_ins_ba1_recon, sample_trans_com_ba2_recon, identity_a[i]))




        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        ori_a, ori_b = torch.cat(ori_a), torch.cat(ori_b)
        ins_a_recon, ins_b_recon = torch.cat(ins_a_recon), torch.cat(ins_b_recon)
        ins_ba1, ins_ba2 = torch.cat(ins_ba1), torch.cat(ins_ba2)
        ins_ab1, ins_ab2 = torch.cat(ins_ab1), torch.cat(ins_ab2)

        trans_com_ab1_ins_ab1,trans_com_ab1_ins_ab2,trans_com_ab2_ins_ab1,trans_com_ab2_ins_ab2=\
            torch.cat(trans_com_ab1_ins_ab1), torch.cat(trans_com_ab1_ins_ab2),torch.cat(trans_com_ab2_ins_ab1), torch.cat(trans_com_ab2_ins_ab2)
        trans_com_ba1_ins_ba1,trans_com_ba1_ins_ba2,trans_com_ba2_ins_ba1,trans_com_ba2_ins_ba2=\
            torch.cat(trans_com_ba1_ins_ba1), torch.cat(trans_com_ba1_ins_ba2),torch.cat(trans_com_ba2_ins_ba1), torch.cat(trans_com_ba2_ins_ba2)

        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2,\
               ins_a,ins_a_recon,ins_ab1, ins_ab2,ins_b,ins_b_recon, ins_ba1, ins_ba2, \
               ori_a,trans_com_ab1_ins_ab1,trans_com_ab1_ins_ab2,trans_com_ab2_ins_ab1,trans_com_ab2_ins_ab2,\
               ori_b,trans_com_ba1_ins_ba1,trans_com_ba1_ins_ba2,trans_com_ba2_ins_ba1,trans_com_ba2_ins_ba2



    def com_dis_update(self, x_a, x_b, hyperparameters):
        self.com_dis_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, _ = self.gen_a.encode(x_a)
        c_b, _ = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.com_dis_opt.step()

    def ins_dis_update(self, ins_a, ins_b, hyperparameters):
        self.ins_dis_opt.zero_grad()
        s_ins_a = Variable(torch.randn(ins_a.size(0), self.style_dim, 1, 1).cuda())
        s_ins_b = Variable(torch.randn(ins_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_ins_a, _ = self.gen_a.encode(ins_a)
        c_ins_b, _ = self.gen_b.encode(ins_b)
        # decode (cross domain)
        ins_ba = self.gen_a.decode(c_ins_b, s_ins_a)
        ins_ab = self.gen_b.decode(c_ins_a, s_ins_b)
        # D loss
        self.loss_dis_ins_a = self.dis_a.calc_dis_loss(ins_ba.detach(), ins_a)
        self.loss_dis_ins_b = self.dis_b.calc_dis_loss(ins_ab.detach(), ins_b)
        self.loss_dis_ins_total = hyperparameters['gan_w'] * self.loss_dis_ins_a + hyperparameters['gan_w'] * self.loss_dis_ins_b
        self.loss_dis_ins_total.backward()
        self.ins_dis_opt.step()

    def update_learning_rate(self, end):
        if end:
            if self.com_dis_scheduler is not None:
                self.com_dis_scheduler.step()
            if self.ins_dis_scheduler is not None:
                self.ins_dis_scheduler.step()
            if self.com_gen_scheduler is not None:
                self.com_gen_scheduler.step()
            if self.ins_gen_scheduler is not None:
                self.ins_gen_scheduler.step()
        else:
            if self.com_dis_scheduler is not None:
                self.com_dis_scheduler.step()
            if self.ins_dis_scheduler is not None:
                self.ins_dis_scheduler.step()
            if self.com_gen_scheduler is not None:
                self.com_gen_scheduler.step()
            if self.ins_gen_scheduler is not None:
                self.ins_gen_scheduler.step()

    def resume(self, snapshot_dir, hyperparameters):
        if hyperparameters['end-to-end']:
            last_save = get_model_list(snapshot_dir, "gen")
            # print(last_save)
            state = torch.load(last_save)

            ##loading generator networks
            self.gen_a.load_state_dict(state['bgr'])
            self.gen_b.load_state_dict(state['tar'])
            self.gen_ins_a.load_state_dict(state['ins_a'])
            self.gen_ins_b.load_state_dict(state['ins_b'])
            iterations = int(last_save[-11:-3])
            #
            # ##loading discriminator networks
            # last_save=get_model_list(snapshot_dir,'dis')
            # state=torch.load(last_save)
            # self.dis_a.load_state_dict(state['bgr'])
            # self.dis_b.load_state_dict(state['tar'])
            # self.dis_ins_a.load_state_dict(state['ins_a'])
            # self.dis_ins_a.load_state_dict(state['ins_b'])
            #
            # ##loading optimizer
            # state=torch.load(os.path.join(snapshot_dir,'opt.pt'))
            # self.gen_opt.load_state_dict(state['gen'])
            # self.dis_opt.load_state_dict(state['dis'])
            #
            # ##re-create scheduler
            # self.gen_scheduler=get_scheduler(self.gen_opt,hyperparameters,iterations)
            # self.dis_scheduler=get_scheduler(self.dis_opt,hyperparameters,iterations)
            last_save = get_model_list(snapshot_dir, "gen")
            # print(last_save)
            state = torch.load(last_save)

            ##loading generator networks
            self.gen_a.load_state_dict(state['bgr'])
            self.gen_b.load_state_dict(state['tar'])
            self.gen_ins_a.load_state_dict(state['ins_a'])
            self.gen_ins_b.load_state_dict(state['ins_b'])
            iterations = int(last_save[-11:-3])

            ##loading discriminator networks
            last_save = get_model_list(snapshot_dir, 'dis')
            state = torch.load(last_save)
            self.dis_a.load_state_dict(state['bgr'])
            self.dis_b.load_state_dict(state['tar'])
            self.dis_ins_a.load_state_dict(state['ins_a'])
            self.dis_ins_a.load_state_dict(state['ins_b'])

            ##loading optimizer
            state = torch.load(os.path.join(snapshot_dir, 'opt.pt'))
            self.com_gen_opt.load_state_dict(state['com_gen'])
            self.ins_gen_opt.load_state_dict(state['ins_gen'])
            self.com_dis_opt.load_state_dict(state['com_dis'])
            self.ins_dis_opt.load_state_dict(state['ins_dis'])

            ##re-create scheduler
            self.com_gen_scheduler = get_scheduler(self.com_gen_opt, hyperparameters, iterations)
            self.ins_gen_scheduler = get_scheduler(self.ins_gen_opt, hyperparameters, iterations)
            self.com_dis_scheduler = get_scheduler(self.com_dis_opt, hyperparameters, iterations)
            self.ins_dis_scheduler = get_scheduler(self.ins_dis_opt, hyperparameters, iterations)
        else:
            last_save = get_model_list(snapshot_dir, "gen")
            # print(last_save)
            state = torch.load(last_save)

            ##loading generator networks
            self.gen_a.load_state_dict(state['bgr'])
            self.gen_b.load_state_dict(state['tar'])
            self.gen_ins_a.load_state_dict(state['ins_a'])
            self.gen_ins_b.load_state_dict(state['ins_b'])
            iterations = int(last_save[-11:-3])

            ##loading discriminator networks
            last_save = get_model_list(snapshot_dir, 'dis')
            state = torch.load(last_save)
            self.dis_a.load_state_dict(state['bgr'])
            self.dis_b.load_state_dict(state['tar'])
            self.dis_ins_a.load_state_dict(state['ins_a'])
            self.dis_ins_a.load_state_dict(state['ins_b'])

            ##loading optimizer
            state = torch.load(os.path.join(snapshot_dir, 'opt.pt'))
            self.com_gen_opt.load_state_dict(state['com_gen'])
            self.ins_gen_opt.load_state_dict(state['ins_gen'])
            self.com_dis_opt.load_state_dict(state['com_dis'])
            self.ins_dis_opt.load_state_dict(state['ins_dis'])

            ##re-create scheduler
            self.com_gen_scheduler = get_scheduler(self.com_gen_opt, hyperparameters, iterations)
            self.ins_gen_scheduler = get_scheduler(self.ins_gen_opt, hyperparameters, iterations)
            self.com_dis_scheduler = get_scheduler(self.com_dis_opt, hyperparameters, iterations)
            self.ins_dis_scheduler = get_scheduler(self.ins_dis_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations, end):
        if end:
            snap_gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
            snap_dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
            snap_opt_name = os.path.join(snapshot_dir, 'opt.pt')

            # torch.save({'bgr': self.gen_a.state_dict(), 'tar': self.gen_b.state_dict(), 'ins_a': self.gen_ins_a.state_dict(),
            #             'ins_b': self.gen_ins_b.state_dict()}, snap_gen_name)
            # torch.save({'bgr': self.dis_a.state_dict(), 'tar': self.dis_b.state_dict(), 'ins_a': self.dis_ins_a.state_dict(),
            #             'ins_b': self.dis_ins_b.state_dict()}, snap_dis_name)
            #
            # torch.save({'gen': self.gen_opt.state_dict(),
            #             'dis': self.dis_opt.state_dict(),}, snap_opt_name)
            torch.save(
                {'bgr': self.gen_a.state_dict(), 'tar': self.gen_b.state_dict(), 'ins_a': self.gen_ins_a.state_dict(),
                 'ins_b': self.gen_ins_b.state_dict()}, snap_gen_name)
            torch.save(
                {'bgr': self.dis_a.state_dict(), 'tar': self.dis_b.state_dict(), 'ins_a': self.dis_ins_a.state_dict(),
                 'ins_b': self.dis_ins_b.state_dict()}, snap_dis_name)
            torch.save({'com_gen': self.com_gen_opt.state_dict(),
                        'ins_gen': self.ins_gen_opt.state_dict(),
                        'com_dis': self.com_dis_opt.state_dict(),
                        'ins_dis': self.ins_dis_opt.state_dict()}, snap_opt_name)
        else:
            snap_gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
            snap_dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
            snap_opt_name = os.path.join(snapshot_dir, 'opt.pt')

            torch.save(
                {'bgr': self.gen_a.state_dict(), 'tar': self.gen_b.state_dict(), 'ins_a': self.gen_ins_a.state_dict(),
                 'ins_b': self.gen_ins_b.state_dict()}, snap_gen_name)
            torch.save(
                {'bgr': self.dis_a.state_dict(), 'tar': self.dis_b.state_dict(), 'ins_a': self.dis_ins_a.state_dict(),
                 'ins_b': self.dis_ins_b.state_dict()}, snap_dis_name)
            torch.save({'com_gen': self.com_gen_opt.state_dict(),
                        'ins_gen': self.ins_gen_opt.state_dict(),
                        'com_dis': self.com_dis_opt.state_dict(),
                        'ins_dis': self.ins_dis_opt.state_dict()}, snap_opt_name)

    def pad(self, ins, background, label):

        background = background.cpu().numpy()
        ins = ins.cpu().numpy()
        ymin = int(label[1] - 1)
        ymax = int(label[3] - 1)
        xmin = int(label[0] - 1)
        xmax = int(label[2] - 1)
        # print(background.shape)
        # print(ins.shape)
        result = np.zeros(background.shape)
        padding = np.transpose(background[0], (1, 2, 0))
        padding = cv2.resize(padding, (256, 256))
        shape = padding[ymin:ymax, xmin:xmax, :].shape
        instance_a = np.transpose(ins[0], (1, 2, 0))
        ins_a = cv2.resize(instance_a, (shape[1], shape[0]))
        padding[ymin:ymax, xmin:xmax, :] = ins_a
        result[0] = np.transpose(padding, (2, 0, 1))
        result = torch.from_numpy(result)
        # print(padding.shape)
        return result

    def end_pad(self, ins, background, label):

        background = background.detach().cpu().numpy()
        ins = ins.detach().cpu().numpy()
        ymin = int(label[1] - 1)
        ymax = int(label[3] - 1)
        xmin = int(label[0] - 1)
        xmax = int(label[2] - 1)
        # print(background.shape)
        # print(ins.shape)
        result = np.zeros(background.shape)
        padding = np.transpose(background[0], (1, 2, 0))
        padding = cv2.resize(padding, (256, 256))
        shape = padding[ymin:ymax, xmin:xmax, :].shape
        instance_a = np.transpose(ins[0], (1, 2, 0))
        ins_a = cv2.resize(instance_a, (shape[1], shape[0]))
        padding[ymin:ymax, xmin:xmax, :] = ins_a
        result[0] = np.transpose(padding, (2, 0, 1))
        result = torch.from_numpy(result).float().cuda()
        # print(padding.shape)
        return result

    def ori_pad(self, ins, background, label):

        background = background.cpu().numpy()
        ins = ins.cpu().numpy()
        ymin = int(label[1] - 1)
        ymax = int(label[3] - 1)
        xmin = int(label[0] - 1)
        xmax = int(label[2] - 1)
        padding = np.transpose(background, (1, 2, 0))
        # padding = cv2.resize(padding, (256, 256))
        shape = padding[ymin:ymax, xmin:xmax, :].shape
        # print(shape)
        # cv2.imwrite('/root/UNIT/results/test_pad/'+str(index)+'_back.jpg',padding)
        instance_a = np.transpose(ins, (1, 2, 0))
        ins_a = cv2.resize(instance_a, (shape[1], shape[0]))
        # cv2.imwrite('/root/UNIT/results/test_pad/' + str(index) + '_ins.jpg', ins_a)
        padding[ymin:ymax, xmin:xmax, :] = ins_a
        # cv2.imwrite('/root/UNIT/results/test_pad/' + str(index) + '_pad.jpg', padding)
        padding = np.transpose(padding, (2, 0, 1))
        shape = padding.shape
        result = np.zeros((1, shape[0], shape[1], shape[2]))
        result[0] = padding
        result = torch.from_numpy(result)
        # print(padding.shape)
        return result


class UNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(UNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = VAEGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.gen_ins_a=VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'])
        self.gen_ins_b = VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'])

        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.dis_ins_a=Ins_Discriminator(hyperparameters['Ins_Dis'],3)
        self.dis_ins_b = Ins_Discriminator(hyperparameters['Ins_Dis'], 3)

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        if hyperparameters['end-to-end']:
            # entire_dis_params=list(self.dis_a.parameters()) + list(self.dis_b.parameters())+list(self.dis_ins_a.parameters())+list(self.dis_ins_b.parameters())
            # entire_gen_params=list(self.gen_a.parameters()) + list(self.gen_b.parameters())+list(self.gen_ins_a.parameters())+list(self.gen_ins_b.parameters())
            # self.dis_opt = torch.optim.Adam([p for p in entire_dis_params if p.requires_grad],
            #                                 lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
            # self.gen_opt = torch.optim.Adam([p for p in entire_gen_params if p.requires_grad],
            #                                 lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
            # self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
            # self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

            com_dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
            com_gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
            ins_dis_params = list(self.dis_ins_a.parameters())+list(self.dis_ins_b.parameters())
            ins_gen_params = list(self.gen_ins_a.parameters())+list(self.gen_ins_b.parameters())


            self.com_dis_opt = torch.optim.Adam([p for p in com_dis_params if p.requires_grad],
                                            lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
            self.com_gen_opt = torch.optim.Adam([p for p in com_gen_params if p.requires_grad],
                                            lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])


            self.ins_dis_opt=torch.optim.Adam([p for p in ins_dis_params if p.requires_grad],
                                            lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

            self.ins_gen_opt=torch.optim.Adam([p for p in ins_gen_params if p.requires_grad],
                                            lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

            self.com_dis_scheduler = get_scheduler(self.com_dis_opt, hyperparameters)
            self.com_gen_scheduler = get_scheduler(self.com_gen_opt, hyperparameters)
            self.ins_dis_scheduler = get_scheduler(self.ins_dis_opt,hyperparameters)
            self.ins_gen_scheduler = get_scheduler(self.ins_gen_opt,hyperparameters)

        else:
            com_dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
            com_gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
            ins_dis_params = list(self.dis_ins_a.parameters())+list(self.dis_ins_b.parameters())
            ins_gen_params = list(self.gen_ins_a.parameters())+list(self.gen_ins_b.parameters())


            self.com_dis_opt = torch.optim.Adam([p for p in com_dis_params if p.requires_grad],
                                            lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
            self.com_gen_opt = torch.optim.Adam([p for p in com_gen_params if p.requires_grad],
                                            lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])


            self.ins_dis_opt=torch.optim.Adam([p for p in ins_dis_params if p.requires_grad],
                                            lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

            self.ins_gen_opt=torch.optim.Adam([p for p in ins_gen_params if p.requires_grad],
                                            lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

            self.com_dis_scheduler = get_scheduler(self.com_dis_opt, hyperparameters)
            self.com_gen_scheduler = get_scheduler(self.com_gen_opt, hyperparameters)
            self.ins_dis_scheduler = get_scheduler(self.ins_dis_opt,hyperparameters)
            self.ins_gen_scheduler = get_scheduler(self.ins_gen_opt,hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))
        self.dis_ins_a.apply(weights_init('gaussian'))
        self.dis_ins_b.apply(weights_init('gaussian'))
        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def Gen_update(self,x1,x2,y1,y2,cfg):
        #print("it's time for common generative network updating!")
        self.com_gen_update(x1,y1,cfg)
        #print("it's time for Ins generative network updating!")
        self.Ins_gen_update(x2,y2)

    def Dis_update(self, x1, x2, y1,y2, cfg):
        #print("it's time for common dis network updating!")
        self.com_dis_update(x1, y1, cfg)
        #print("it's time for ins dis network updating!")
        #print(x2.shape)
        #print(y2.shape)
        self.Ins_dis_update(x2,y2, cfg)

    def End_mode_Dis_update(self,x_a,Ins_a,x_b,Ins_b,index,hyperparameters,smooth_a,smooth_b):
        #print(smooth_a)
        # smooth_a=Variable(smooth_a, requires_grad=False)
        # smooth_b=Variable(smooth_b, requires_grad=False)
        self.ins_dis_opt.zero_grad()
        Ins_a_latent,noise_a=self.gen_ins_a.encode(Ins_a)
        Ins_a_recon=self.gen_ins_a.decode(Ins_a_latent+noise_a)

        Ins_b_latent,noise_b=self.gen_ins_b.encode(Ins_b)
        Ins_b_recon=self.gen_ins_b.decode(Ins_b_latent+noise_b)
        #print(Ins_b_recon.shape)
        self.Ins_dis_loss=hyperparameters['Dis_loss_weight']*self.dis_ins_a.Dis_loss(Ins_a_recon,Ins_a)+hyperparameters['Dis_loss_weight']*self.dis_ins_b.Dis_loss(Ins_b_recon,Ins_b)
        self.Ins_dis_loss.backward()

        self.com_dis_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)

        h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(h_a_recon + n_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(h_b_recon + n_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        x_ba=self.end_pad(Ins_a_recon,x_ba,identity_a[index])
        x_ab=self.end_pad(Ins_b_recon,x_ab,identity_b[index])

        x_aba=self.end_pad(Ins_a_recon,x_aba,identity_a[index])
        x_bab=self.end_pad(Ins_b_recon,x_bab,identity_b[index])

        # D loss
        #x_ba.requires_grad_()
        #x_ba = x_ba.float()
        #print(type(x_ba))
        self.loss_dis_a = self.dis_a.smooth_dis_loss(x_ba.detach(), x_a,smooth_a)
        self.loss_dis_b = self.dis_b.smooth_dis_loss(x_ab.detach(), x_b,smooth_b)
        self.loss_dis_aba = self.dis_a.smooth_dis_loss(x_bab.detach(), x_b,smooth_b)
        self.loss_dis_bab = self.dis_b.smooth_dis_loss(x_aba.detach(), x_a,smooth_a)
        # self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        # self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b+ \
                              hyperparameters['gan_w'] * self.loss_dis_aba + hyperparameters['gan_w'] * self.loss_dis_bab
        self.loss_dis_total.backward()
        self.com_dis_opt.step()

    def End_mode_Gen_update(self,x_a,Ins_a,x_b,Ins_b,index,hyperparameters):
        #print("start ins gen update")
        self.ins_gen_opt.zero_grad()

        Ins_a_latent, noise_Ins_a=self.gen_ins_a.encode(Ins_a)
        Ins_a_recon=self.gen_ins_a.decode(Ins_a_latent+noise_Ins_a)

        self.InsGen_a_recon_loss=self.recon_criterion(Ins_a_recon,Ins_a)
        self.InsGen_a_kl_loss=self.__compute_kl(Ins_a_latent)

        Ins_b_latent, noise_Ins_b=self.gen_ins_b.encode(Ins_b)
        Ins_b_recon=self.gen_ins_b.decode(Ins_b_latent+noise_Ins_b)

        self.InsGen_b_recon_loss=self.recon_criterion(Ins_b_recon,Ins_b)
        self.InsGen_b_kl_loss=self.__compute_kl(Ins_b_latent)

        self.InsGen_loss=self.InsGen_a_kl_loss+self.InsGen_a_recon_loss+self.InsGen_b_kl_loss+self.InsGen_b_recon_loss

        self.InsGen_loss.backward()
        self.ins_gen_opt.step()


        self.com_gen_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(h_a + n_a)
        x_b_recon = self.gen_b.decode(h_b + n_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # encode again
        h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(h_a_recon + n_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(h_b_recon + n_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        x_a_recon=self.end_pad(Ins_a_recon,x_a_recon,identity_a[index])
        x_b_recon=self.end_pad(Ins_b_recon,x_b_recon,identity_b[index])

        x_ab=self.end_pad(Ins_b_recon,x_ab,identity_b[index])
        x_ba=self.end_pad(Ins_a_recon,x_ba,identity_a[index])

        x_aba=self.end_pad(Ins_a_recon,x_aba,identity_a[index])
        x_bab=self.end_pad(Ins_b_recon,x_bab,identity_b[index])


        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b
        self.loss_gen_total.backward()
        self.com_gen_opt.step()




    def forward(self, x_a, x_b):
        self.eval()
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def com_gen_update(self, x_a, x_b, hyperparameters):
        self.com_gen_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(h_a + n_a)
        x_b_recon = self.gen_b.decode(h_b + n_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # encode again
        h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(h_a_recon + n_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(h_b_recon + n_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b
        self.loss_gen_total.backward()
        self.com_gen_opt.step()



    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b,ins_a,ins_b,mode='train'):
        self.eval()
        #print(mode)
        save_name = "./Data/" + str(mode) + "_a_identity_2.npy"
        identity_a = np.load(save_name)
        #print(save_name)
        save_name = "./Data/" + str(mode) + "_b_identity_2.npy"
        identity_b = np.load(save_name)
        #print(save_name)
        original_path = "./Data/result_outputs/" + mode + ''
        isExist = os.path.exists(original_path)
        if not isExist:
            os.mkdir(original_path)
        x_a_recon, x_b_recon, x_ba, x_ab,ins_a_recon,ins_b_recon = [], [], [], [],[],[]
        ori_a,ori_b,recon_a,recon_b,trans_a,trans_b=[],[],[],[],[],[]
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            h_ins_a,_=self.gen_ins_a.encode(ins_a[i].unsqueeze(0))
            h_ins_b, _ = self.gen_ins_b.encode(ins_b[i].unsqueeze(0))
            sample_com_a_recon=self.gen_a.decode(h_a)
            sample_com_b_recon = self.gen_b.decode(h_b)
            sample_ins_a_recon = self.gen_ins_a.decode(h_ins_a)
            sample_ins_b_recon = self.gen_ins_b.decode(h_ins_b)
            sample_trans_a_recon = self.gen_a.decode(h_b)
            sample_trans_b_recon = self.gen_b.decode(h_a)
            #print(sample_com_b_recon.shape)

            x_a_recon.append(sample_com_a_recon)
            x_b_recon.append(sample_com_b_recon)
            ins_a_recon.append(sample_ins_a_recon)
            ins_b_recon.append(sample_ins_b_recon)
            x_ba.append(sample_trans_a_recon)
            x_ab.append(sample_trans_b_recon)
            #print(self.ori_pad(ins_a[i],x_a[i],identity_a[i]).shape)
            ori_a.append(self.ori_pad(ins_a[i],x_a[i],identity_a[i]))
            ori_b.append(self.ori_pad(ins_b[i],x_b[i], identity_b[i]))
            #print(sample_ins_a_recon.shape)
            #print("sample")
            recon_a.append(self.pad(sample_ins_a_recon, sample_com_a_recon, identity_a[i]))
            recon_b.append(self.pad(sample_ins_b_recon, sample_com_b_recon, identity_b[i]))
            trans_a.append(self.pad(sample_ins_a_recon, sample_trans_a_recon, identity_a[i]))
            trans_b.append(self.pad(sample_ins_b_recon, sample_trans_b_recon, identity_b[i]))


        x_a_recon, x_b_recon,ins_a_recon,ins_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon),torch.cat(ins_a_recon),torch.cat(ins_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)

        ori_a,ori_b,recon_a,recon_b,trans_a,trans_b=torch.cat(ori_a),torch.cat(ori_b),torch.cat(recon_a),torch.cat(recon_b),torch.cat(trans_a),torch.cat(trans_b)

        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba,ins_a,ins_a_recon,ins_b,ins_b_recon,ori_a,ori_b,recon_a,recon_b,trans_a,trans_b

    def com_dis_update(self, x_a, x_b, hyperparameters):
        self.com_dis_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.com_dis_opt.step()

    def Ins_gen_update(self,Ins_a,Ins_b):
        #print("start ins gen update")
        self.ins_gen_opt.zero_grad()

        Ins_a_latent, noise_Ins_a=self.gen_ins_a.encode(Ins_a)
        Ins_a_recon=self.gen_ins_a.decode(Ins_a_latent+noise_Ins_a)

        self.InsGen_a_recon_loss=self.recon_criterion(Ins_a_recon,Ins_a)
        self.InsGen_a_kl_loss=self.__compute_kl(Ins_a_latent)

        Ins_b_latent, noise_Ins_b=self.gen_ins_b.encode(Ins_b)
        Ins_b_recon=self.gen_ins_b.decode(Ins_b_latent+noise_Ins_b)

        self.InsGen_b_recon_loss=self.recon_criterion(Ins_b_recon,Ins_b)
        self.InsGen_b_kl_loss=self.__compute_kl(Ins_b_latent)

        self.InsGen_loss=self.InsGen_a_kl_loss+self.InsGen_a_recon_loss+self.InsGen_b_kl_loss+self.InsGen_b_recon_loss
        self.InsGen_loss.backward()
        self.ins_gen_opt.step()

    def Ins_dis_update(self,Ins_a,Ins_b,cfg):
        #print("hello ins dis update")
        self.ins_dis_opt.zero_grad()
        Ins_a_latent,noise_a=self.gen_ins_a.encode(Ins_a)
        Ins_a_recon=self.gen_ins_a.decode(Ins_a_latent+noise_a)
        #print(Ins_a_recon.shape)
        #print("ins_a")
        Ins_b_latent,noise_b=self.gen_ins_b.encode(Ins_b)
        Ins_b_recon=self.gen_ins_b.decode(Ins_b_latent+noise_b)
        #print(Ins_b_recon.shape)
        self.Ins_dis_loss=cfg['Dis_loss_weight']*self.dis_ins_a.Dis_loss(Ins_a_recon,Ins_a)+cfg['Dis_loss_weight']*self.dis_ins_b.Dis_loss(Ins_b_recon,Ins_b)
        self.Ins_dis_loss.backward()
        self.ins_dis_opt.step()

    def update_learning_rate(self,end):
        if end:
            if self.com_dis_scheduler is not None:
                self.com_dis_scheduler.step()
            if self.ins_dis_scheduler is not None:
                self.ins_dis_scheduler.step()
            if self.com_gen_scheduler is not None:
                self.com_gen_scheduler.step()
            if self.ins_gen_scheduler is not None:
                self.ins_gen_scheduler.step()
        else:
            if self.com_dis_scheduler is not None:
                self.com_dis_scheduler.step()
            if self.ins_dis_scheduler is not None:
                self.ins_dis_scheduler.step()
            if self.com_gen_scheduler is not None:
                self.com_gen_scheduler.step()
            if self.ins_gen_scheduler is not None:
                self.ins_gen_scheduler.step()

    def resume(self, snapshot_dir, hyperparameters):
        if hyperparameters['end-to-end']:
            last_save=get_model_list(snapshot_dir,"gen")
            #print(last_save)
            state=torch.load(last_save)

            ##loading generator networks
            self.gen_a.load_state_dict(state['bgr'])
            self.gen_b.load_state_dict(state['tar'])
            self.gen_ins_a.load_state_dict(state['ins_a'])
            self.gen_ins_b.load_state_dict(state['ins_b'])
            iterations = int(last_save[-11:-3])
            #
            # ##loading discriminator networks
            # last_save=get_model_list(snapshot_dir,'dis')
            # state=torch.load(last_save)
            # self.dis_a.load_state_dict(state['bgr'])
            # self.dis_b.load_state_dict(state['tar'])
            # self.dis_ins_a.load_state_dict(state['ins_a'])
            # self.dis_ins_a.load_state_dict(state['ins_b'])
            #
            # ##loading optimizer
            # state=torch.load(os.path.join(snapshot_dir,'opt.pt'))
            # self.gen_opt.load_state_dict(state['gen'])
            # self.dis_opt.load_state_dict(state['dis'])
            #
            # ##re-create scheduler
            # self.gen_scheduler=get_scheduler(self.gen_opt,hyperparameters,iterations)
            # self.dis_scheduler=get_scheduler(self.dis_opt,hyperparameters,iterations)
            last_save=get_model_list(snapshot_dir,"gen")
            #print(last_save)
            state=torch.load(last_save)

            ##loading generator networks
            self.gen_a.load_state_dict(state['bgr'])
            self.gen_b.load_state_dict(state['tar'])
            self.gen_ins_a.load_state_dict(state['ins_a'])
            self.gen_ins_b.load_state_dict(state['ins_b'])
            iterations = int(last_save[-11:-3])

            ##loading discriminator networks
            last_save=get_model_list(snapshot_dir,'dis')
            state=torch.load(last_save)
            self.dis_a.load_state_dict(state['bgr'])
            self.dis_b.load_state_dict(state['tar'])
            self.dis_ins_a.load_state_dict(state['ins_a'])
            self.dis_ins_a.load_state_dict(state['ins_b'])

            ##loading optimizer
            state=torch.load(os.path.join(snapshot_dir,'opt.pt'))
            self.com_gen_opt.load_state_dict(state['com_gen'])
            self.ins_gen_opt.load_state_dict(state['ins_gen'])
            self.com_dis_opt.load_state_dict(state['com_dis'])
            self.ins_dis_opt.load_state_dict(state['ins_dis'])

            ##re-create scheduler
            self.com_gen_scheduler=get_scheduler(self.com_gen_opt,hyperparameters,iterations)
            self.ins_gen_scheduler=get_scheduler(self.ins_gen_opt,hyperparameters,iterations)
            self.com_dis_scheduler=get_scheduler(self.com_dis_opt,hyperparameters,iterations)
            self.ins_dis_scheduler=get_scheduler(self.ins_dis_opt,hyperparameters,iterations)
        else:
            last_save=get_model_list(snapshot_dir,"gen")
            #print(last_save)
            state=torch.load(last_save)

            ##loading generator networks
            self.gen_a.load_state_dict(state['bgr'])
            self.gen_b.load_state_dict(state['tar'])
            self.gen_ins_a.load_state_dict(state['ins_a'])
            self.gen_ins_b.load_state_dict(state['ins_b'])
            iterations = int(last_save[-11:-3])

            ##loading discriminator networks
            last_save=get_model_list(snapshot_dir,'dis')
            state=torch.load(last_save)
            self.dis_a.load_state_dict(state['bgr'])
            self.dis_b.load_state_dict(state['tar'])
            self.dis_ins_a.load_state_dict(state['ins_a'])
            self.dis_ins_a.load_state_dict(state['ins_b'])

            ##loading optimizer
            state=torch.load(os.path.join(snapshot_dir,'opt.pt'))
            self.com_gen_opt.load_state_dict(state['com_gen'])
            self.ins_gen_opt.load_state_dict(state['ins_gen'])
            self.com_dis_opt.load_state_dict(state['com_dis'])
            self.ins_dis_opt.load_state_dict(state['ins_dis'])

            ##re-create scheduler
            self.com_gen_scheduler=get_scheduler(self.com_gen_opt,hyperparameters,iterations)
            self.ins_gen_scheduler=get_scheduler(self.ins_gen_opt,hyperparameters,iterations)
            self.com_dis_scheduler=get_scheduler(self.com_dis_opt,hyperparameters,iterations)
            self.ins_dis_scheduler=get_scheduler(self.ins_dis_opt,hyperparameters,iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations,end):
        if end:
            snap_gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
            snap_dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
            snap_opt_name = os.path.join(snapshot_dir, 'opt.pt')

            # torch.save({'bgr': self.gen_a.state_dict(), 'tar': self.gen_b.state_dict(), 'ins_a': self.gen_ins_a.state_dict(),
            #             'ins_b': self.gen_ins_b.state_dict()}, snap_gen_name)
            # torch.save({'bgr': self.dis_a.state_dict(), 'tar': self.dis_b.state_dict(), 'ins_a': self.dis_ins_a.state_dict(),
            #             'ins_b': self.dis_ins_b.state_dict()}, snap_dis_name)
            #
            # torch.save({'gen': self.gen_opt.state_dict(),
            #             'dis': self.dis_opt.state_dict(),}, snap_opt_name)
            torch.save({'bgr':self.gen_a.state_dict(), 'tar':self.gen_b.state_dict(),'ins_a':self.gen_ins_a.state_dict(),'ins_b':self.gen_ins_b.state_dict()},snap_gen_name)
            torch.save({'bgr':self.dis_a.state_dict(), 'tar':self.dis_b.state_dict(),'ins_a':self.dis_ins_a.state_dict(),'ins_b':self.dis_ins_b.state_dict()},snap_dis_name)
            torch.save({'com_gen':self.com_gen_opt.state_dict(),
                       'ins_gen':self.ins_gen_opt.state_dict(),
                       'com_dis':self.com_dis_opt.state_dict(),
                       'ins_dis':self.ins_dis_opt.state_dict()},snap_opt_name)
        else:
            snap_gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
            snap_dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
            snap_opt_name = os.path.join(snapshot_dir, 'opt.pt')

            torch.save({'bgr':self.gen_a.state_dict(), 'tar':self.gen_b.state_dict(),'ins_a':self.gen_ins_a.state_dict(),'ins_b':self.gen_ins_b.state_dict()},snap_gen_name)
            torch.save({'bgr':self.dis_a.state_dict(), 'tar':self.dis_b.state_dict(),'ins_a':self.dis_ins_a.state_dict(),'ins_b':self.dis_ins_b.state_dict()},snap_dis_name)
            torch.save({'com_gen':self.com_gen_opt.state_dict(),
                       'ins_gen':self.ins_gen_opt.state_dict(),
                       'com_dis':self.com_dis_opt.state_dict(),
                       'ins_dis':self.ins_dis_opt.state_dict()},snap_opt_name)

    def pad(self,ins, background,label):

        background = background.cpu().numpy()
        ins = ins.cpu().numpy()
        ymin = int(label[1] - 1)
        ymax = int(label[3] - 1)
        xmin = int(label[0] - 1)
        xmax = int(label[2] - 1)
        #print(background.shape)
        #print(ins.shape)
        result=np.zeros(background.shape)
        padding = np.transpose(background[0], (1, 2, 0))
        padding = cv2.resize(padding, (256, 256))
        shape = padding[ymin:ymax, xmin:xmax, :].shape
        instance_a = np.transpose(ins[0], (1, 2, 0))
        ins_a = cv2.resize(instance_a, (shape[1], shape[0]))
        padding[ymin:ymax, xmin:xmax, :] = ins_a
        result[0]=np.transpose(padding,(2,0,1))
        result = torch.from_numpy(result)
        #print(padding.shape)
        return result

    def end_pad(self,ins, background,label):

        background = background.detach().cpu().numpy()
        ins = ins.detach().cpu().numpy()
        ymin = int(label[1] - 1)
        ymax = int(label[3] - 1)
        xmin = int(label[0] - 1)
        xmax = int(label[2] - 1)
        #print(background.shape)
        #print(ins.shape)
        result=np.zeros(background.shape)
        padding = np.transpose(background[0], (1, 2, 0))
        padding = cv2.resize(padding, (256, 256))
        shape = padding[ymin:ymax, xmin:xmax, :].shape
        instance_a = np.transpose(ins[0], (1, 2, 0))
        ins_a = cv2.resize(instance_a, (shape[1], shape[0]))
        padding[ymin:ymax, xmin:xmax, :] = ins_a
        result[0]=np.transpose(padding,(2,0,1))
        result = torch.from_numpy(result).float().cuda()
        #print(padding.shape)
        return result

    def ori_pad(self,ins, background,label):

        background = background.cpu().numpy()
        ins = ins.cpu().numpy()
        ymin = int(label[1] - 1)
        ymax = int(label[3] - 1)
        xmin = int(label[0] - 1)
        xmax = int(label[2] - 1)
        padding = np.transpose(background, (1, 2, 0))
        #padding = cv2.resize(padding, (256, 256))
        shape = padding[ymin:ymax, xmin:xmax, :].shape
        #print(shape)
        #cv2.imwrite('/root/UNIT/results/test_pad/'+str(index)+'_back.jpg',padding)
        instance_a = np.transpose(ins, (1, 2, 0))
        ins_a = cv2.resize(instance_a, (shape[1], shape[0]))
        #cv2.imwrite('/root/UNIT/results/test_pad/' + str(index) + '_ins.jpg', ins_a)
        padding[ymin:ymax, xmin:xmax, :] = ins_a
        #cv2.imwrite('/root/UNIT/results/test_pad/' + str(index) + '_pad.jpg', padding)
        padding=np.transpose(padding, (2, 0, 1))
        shape = padding.shape
        result = np.zeros((1,shape[0],shape[1],shape[2]))
        result[0]=padding
        result = torch.from_numpy(result)
        #print(padding.shape)
        return result