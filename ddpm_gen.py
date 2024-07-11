import copy
import json
import os
import warnings
from absl import app, flags
import random
import shutil
#import cv2

import torch
import numpy as np
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange

from diffusion import *
from model.model import UNet
from score.both import get_inception_and_fid_score
from dataset import ImbalanceCIFAR100, ImbalanceCIFAR10
from score.fid import get_fid_score

FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')


# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 8, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/DDPM_NIH', help='log directory')
flags.DEFINE_string('cond_logdir', './logs/DDPM_NIH', help='cond log directory')
flags.DEFINE_string('uncond_logdir', './logs/DDPM_NIH', help='uncond log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
flags.DEFINE_float('w', 2, help='Guided rate')
# Evaluation
flags.DEFINE_integer('save_step', 5000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_integer('num_images_per_class', 10000, help='the number of generated images for evaluation per class')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')

flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='fid cache')


flags.DEFINE_integer('ckpt_step',-1,help="chekpoint step")
flags.DEFINE_integer('ckpt_step_uncond',-1,help="chekpoint step")
flags.DEFINE_integer('specific_class',-1,help="generate specific class -1 for not utilization")
flags.DEFINE_bool('balanced_dat',False,help="using bal dataset")
flags.DEFINE_integer('ddim_skip_step',10,help="ddim step")
flags.DEFINE_integer('cut_time',1001,help="cut time")
flags.DEFINE_integer('num_class', 10, help='number of class of the pretrained model')
flags.DEFINE_string('sample_method', 'ddim', help='sampling method')

flags.DEFINE_bool('conditional', False, help='conditional generation')
flags.DEFINE_bool('weight', False, help='reweight')
flags.DEFINE_bool('cotrain', False, help='cotrain with an adjusted classifier or not')
flags.DEFINE_bool('logit', False, help='use logit adjustment or not')
flags.DEFINE_bool('augm', False, help='whether to use ADA augmentation')
flags.DEFINE_bool('cfg', False, help='whether to train unconditional generation with with 10\%  probability')

device = torch.device('cuda')

def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y

N_CLASS=10


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))

def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x,y

def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup






def evaluate(sampler, model,save=True,use_eval=True,save_intermediate=False):
    #model.eval()

    with torch.no_grad():
        images = [];labels = [];intermediate_images=[]
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            #Each image corresponds to a random label
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))  
            #change it to corresponding label

            y = torch.randint(FLAGS.num_class, size=(x_T.shape[0], ),device=device)

            batch_images = sampler(x_T.to(device),y,method=FLAGS.sample_method,skip=FLAGS.ddim_skip_step).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy(); #labels = torch.cat(labels, dim=0).cpu().numpy()
    # (IS, IS_std) = get_inception_and_fid_score(
    #     images, mu_cache=FLAGS.fid_mu_cache, sigma_cache=FLAGS.fid_sigma_cache, num_images=FLAGS.num_images,
    #     use_torch=FLAGS.fid_use_torch, verbose=True)

    #np.save(os.path.join(FLAGS.logdir, 'gen_labels_ema_{}_w_{}.npy'.format(FLAGS.ckpt_step,FLAGS.w)), labels)
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    #FID=get_fid_score(mu_cache=FLAGS.fid_mu_cache, sigma_cache=FLAGS.fid_sigma_cache,images=images,num_images=FLAGS.num_images,use_torch=FLAGS.fid_use_torch, verbose=True)
    if save_intermediate:
        return images,intermediate_images

    return (IS, IS_std), FID, images


def eval():
    # model setup
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout,
        cond=FLAGS.conditional, augm=FLAGS.augm, num_class=FLAGS.num_class)

    sampler = GaussianDiffusionSamplerOld(
            model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
             var_type=FLAGS.var_type, w=FLAGS.w,cond = FLAGS.conditional).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    if FLAGS.ckpt_step >= 0:
        ckpt = torch.load(os.path.join(FLAGS.logdir, f'ckpt_{FLAGS.ckpt_step}.pt'))
    else:
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))

    model.load_state_dict(ckpt['net_model'])

    # (IS, IS_std), FID, samples = evaluate(sampler, model)
    # print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    # save_image(
    #    torch.tensor(samples[:256]),
    #    os.path.join(FLAGS.logdir, 'samples.png'),
    #    nrow=16)

    model.load_state_dict(ckpt['ema_model'])
    (IS, IS_std), FID, samples = evaluate(sampler, None)
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, 'samples_ema_{}.png'.format(FLAGS.specific_class)),
        nrow=16)













def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train()
    if FLAGS.eval:
        eval()


    if not FLAGS.train and not FLAGS.eval:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)
