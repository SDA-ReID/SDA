##########################################
# This is the core training code for SDA #
##########################################

import torch
import cv2
import copy
import itertools

from torch.nn import functional as F
from torch import nn
from torch.backends import cudnn

from translate.util.image_pool import ImagePool
from translate.util.util import tensor2im
from translate.models.base_model import BaseModel
from translate.models import networks

from reid.loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy
from reid.utils.meters import AverageMeter
from reid import models
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.data import transforms as T

class SDAModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt, source_classes):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.source_classes = source_classes
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B',
                            'rc', 'reid_ce_B', 'reid_tri_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B']
        visual_names_B = ['real_B', 'fake_A']
        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', '_A', '_B', '_B_ema']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.net_A = models.create(opt.arch, num_features=opt.features, dropout=opt.dropout, num_classes=source_classes)
        self.net_B = models.create(opt.arch, num_features=opt.features, dropout=opt.dropout, num_classes=source_classes+opt.num_clusters)
        self.net_A.cuda()
        self.net_B.cuda()
        self.net_A = nn.DataParallel(self.net_A)
        self.net_B = nn.DataParallel(self.net_B)

        if (opt.init_s):
            initial_weights = load_checkpoint(opt.init_s)
            self.net_A.load_state_dict(initial_weights['state_dict'])
            copy_state_dict(initial_weights['state_dict'], self.net_B)

        self.net_B_ema = copy.deepcopy(self.net_B)

        if self.isTrain:  # define discriminators
            if (opt.netD=='n_layers_proj'):
                assert (opt.gan_mode=='hinge')
                self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netD_B = self.netD_A
            else:
                self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).cuda()  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss().cuda()
            self.criterionIdt = torch.nn.L1Loss().cuda()
            self.criterionMMD = networks.MMD_loss().cuda()
            ### [IMPORTANT] define relation consistency loss ###
            self.criterion_rc = SoftTripletLoss(margin=None, drop=0).cuda()

            # loss functions for ReID
            self.criterion_tri = SoftTripletLoss(margin=0.0, drop=opt.dropout_tri).cuda()

            self.set_optimizer()

    def set_optimizer(self):
        opt = self.opt
        self.set_status()
        self.optimizers = []
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        if (opt.netD=='n_layers_proj'):
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        else:
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        params = []
        for key, value in self.net_B.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": self.opt.lr_reid, "weight_decay": self.opt.weight_decay}]
        self.optimizer_reid = torch.optim.Adam(params)
        self.setup(self.opt)  # reset scheduler

    def set_input(self, source_inputs, target_inputs):
        s_imgs, _, s_pids, _ = source_inputs
        self.real_A = s_imgs.cuda()
        self.label_A = s_pids.cuda()
        t_imgs, _, t_pids, _ = target_inputs
        self.real_B = t_imgs.cuda()
        self.label_B = t_pids.cuda()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.norm_img(self.netG_A(self.real_A))  # G_A(A)
        self.fake_A = self.norm_img(self.netG_B(self.real_B))  # G_B(B)

    def norm_img(self, imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        mean = torch.Tensor(mean).view(1,3,1,1).expand_as(imgs).cuda()
        std = torch.Tensor(std).view(1,3,1,1).expand_as(imgs).cuda()
        return (imgs - mean) / std

    def backward_D_basic(self, netD, real, fake, label=None):
        if (label is not None):
            y = torch.LongTensor(real.size(0)).fill_(label).cuda()
            pred_real = netD(real, y)
            pred_fake = netD(fake.detach(), y)
        else:
            pred_real = netD(real)
            pred_fake = netD(fake.detach())
        # Real
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D.item()

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        if (self.opt.netD=='n_layers_proj'):
            self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, 1)
        else:
            self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        if (self.opt.netD=='n_layers_proj'):
            self.loss_D_B = self.backward_D_basic(self.netD_A, self.real_A, fake_A, 0)
        else:
            self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G_reid_joint(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            idt_A = self.norm_img(self.netG_A(self.real_B))
            loss_idt_A = self.criterionIdt(idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            idt_B = self.norm_img(self.netG_B(self.real_A))
            loss_idt_B = self.criterionIdt(idt_B, self.real_A) * lambda_A * lambda_idt

            del idt_A, idt_B
        else:
            loss_idt_A = 0
            loss_idt_B = 0

        if (self.opt.netD=='n_layers_proj'):
            # GAN loss D_A(G_A(A))
            y = torch.LongTensor(self.fake_B.size(0)).fill_(1).cuda()
            loss_G_A = self.criterionGAN(self.netD_A(self.fake_B, y), True, for_discriminator=False)
            # GAN loss D_B(G_B(B))
            y = torch.LongTensor(self.fake_A.size(0)).fill_(0).cuda()
            loss_G_B = self.criterionGAN(self.netD_A(self.fake_A, y), True, for_discriminator=False)
        else:
            # GAN loss D_A(G_A(A))
            loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True, for_discriminator=False)
            # GAN loss D_B(G_B(B))
            loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True, for_discriminator=False)

        rec_A = self.norm_img(self.netG_B(self.fake_B))   # G_B(G_A(A))
        rec_B = self.norm_img(self.netG_A(self.fake_A))   # G_A(G_B(B))
        # Forward cycle loss || G_B(G_A(A)) - A||
        loss_cycle_A = self.criterionCycle(rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        loss_cycle_B = self.criterionCycle(rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        del rec_A, rec_B

        ### [IMPORTANT] compute relation consisteny loss ###
        real_A_fea, _, _ = self.net_A(self.real_A)
        fake_B_fea_ema, _, _ = self.net_B_ema(self.fake_B)
        loss_rc = self.criterion_rc(fake_B_fea_ema, real_A_fea.detach(), self.label_A)

        del real_A_fea, fake_B_fea_ema

        ### backward SDT ###
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B + loss_rc
        loss_G.backward()


        fake_B_fea, _, fake_B_prob = self.net_B(self.fake_B.detach())
        real_B_fea, _, real_B_prob = self.net_B(self.real_B)

        criterion_ce = CrossEntropyLabelSmooth(self.source_classes+self.opt.num_clusters).cuda()
        loss_reid_ce_B = criterion_ce(torch.cat((fake_B_prob[:,:self.opt.num_clusters+self.source_classes], \
                                                real_B_prob[:,:self.opt.num_clusters+self.source_classes])), \
                                        torch.cat((self.label_A, self.label_B)))
        loss_reid_tri_B = self.criterion_tri(torch.cat((fake_B_fea, real_B_fea)), \
                            torch.cat((fake_B_fea, real_B_fea)), torch.cat((self.label_A, self.label_B)))

        del fake_B_fea, fake_B_prob, real_B_fea, real_B_prob
        ### backward target-domain encoder ###
        loss_reid = loss_reid_ce_B + loss_reid_tri_B
        loss_reid.backward()

        self.loss_G_A = loss_G_A.item()
        self.loss_G_B = loss_G_B.item()
        self.loss_cycle_A = loss_cycle_A.item()
        self.loss_cycle_B = loss_cycle_B.item()
        self.loss_idt_A = loss_idt_A.item()
        self.loss_idt_B = loss_idt_B.item()
        self.loss_rc = loss_rc.item()
        self.loss_reid_ce_B = loss_reid_ce_B.item()
        self.loss_reid_tri_B = loss_reid_tri_B.item()

        del loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, loss_idt_A, loss_idt_B, loss_rc
        del loss_reid_ce_B, loss_reid_tri_B

    def optimize_parameters(self, iter):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        netDs = [self.netD_A, self.netD_B]

        # G_A and G_B
        self.set_requires_grad(netDs, False)  # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.net_B], True)  # Ds require no gradients when optimizing Gs
        self.net_B.module.feat_bn.bias.requires_grad_(False)
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.optimizer_reid.zero_grad()
        self.backward_G_reid_joint()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        self.optimizer_reid.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad(netDs, True)
        self.set_requires_grad([self.net_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate gradients for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        self._update_ema_variables(self.net_B, self.net_B_ema, 0.999, iter)

        self.fake_A = self.fake_A.detach() # for visualization
        self.fake_B = self.fake_B.detach()

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def set_status(self):
        self.netG_A.train()
        self.netG_B.train()
        self.netD_A.train()
        self.netD_B.train()
        self.net_A.eval()
        self.net_B.train()
        self.net_B_ema.train()
        self.set_requires_grad([self.netG_A, self.netG_B, self.netD_A, self.netD_B, self.net_B], True)
        self.set_requires_grad([self.net_A, self.net_B_ema], False)
        self.net_B.module.feat_bn.bias.requires_grad_(False)
