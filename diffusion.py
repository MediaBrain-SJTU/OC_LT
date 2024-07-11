# from dd_code.backdoor.benchmarks.pytorch-ddpm.main import self
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
from tqdm import tqdm
import ipdb
from functools import partial

from torchvision.utils import save_image
import string

class GaussianDiffusionMid(nn.Module):
    def __init__(self,
                 model, beta_1, beta_T, T, dataset,
                 num_class, cfg, cb, tau, weight, finetune,transfer_x0=True,mixing=False,transfer_mode='full'):
        super().__init__()

        self.model = model
        self.T = T
        self.dataset = dataset
        self.num_class = num_class
        self.cfg = cfg
        self.transfer_mode = transfer_mode
        self.cb = cb
        self.tau = tau
        self.weight = weight
        self.finetune = finetune
        self.transfer_x0 = transfer_x0
        self.mixing = mixing
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)


        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.register_buffer(
            'sigma_tsq', 1./alphas_bar-1.)
        self.register_buffer('sigma_t',torch.sqrt(self.sigma_tsq))

    def forward(self, x_0, y_0, augm=None,fix_t=None):
        """
        Algorithm 1.
        """
        # original codes
        if fix_t is None:
            t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        else:
            t = torch.full((x_0.shape[0], ),fix_t)
        noise = torch.randn_like(x_0) 
        ini_noise = noise

        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)

        if self.cfg or self.cb:
            if torch.rand(1)[0] < 1/10:
                y_0 = None
        h,temp_mid = self.model(x_t, t, y=y_0, augm=augm)


        return h,temp_mids


class GaussianDiffusionSamplerOld(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='epsilon', var_type='fixedlarge',w=2,cond=False):
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type
        self.cond = cond
        self.w=w
        print(f"current guidance rate is {w}")
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer(
            'alphas_bar', alphas_bar)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t,
                        method='ddpm',
                        skip=1,
                        eps=None):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        if method == 'ddim':
            assert (eps is not None)
            skip_time = torch.clamp(t - skip, 0, self.T)
            posterior_mean_coef1 = torch.sqrt(extract(self.alphas_bar, t, x_t.shape))
            posterior_mean_coef2 = torch.sqrt(1-extract(self.alphas_bar, t, x_t.shape))
            posterior_mean_coef3 = torch.sqrt(extract(self.alphas_bar, skip_time, x_t.shape))
            posterior_mean_coef4 = torch.sqrt(1-extract(self.alphas_bar, skip_time, x_t.shape))
            posterior_mean = (
                posterior_mean_coef3 / posterior_mean_coef1 * x_t +
                (posterior_mean_coef4 - 
                posterior_mean_coef3 * posterior_mean_coef2 / posterior_mean_coef1) * eps
            )
        else:
            posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)

        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t, y,method, skip):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t, y)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t, t ,y)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            if self.cond:
                eps = self.model(x_t, t ,y)
                eps_g=self.model(x_t, t ,None)
                eps=eps+(self.w)*(eps-eps_g)
                x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
                model_mean, _ = self.q_mean_variance(x_0, x_t, t, method, skip, eps)
            else:
                #ipdb.set_trace()
                eps = self.model(x_t, t)
                x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
                model_mean, _ = self.q_mean_variance(x_0, x_t, t, method, skip, eps)
                #print("un conditional!")
        else:
            raise NotImplementedError(self.mean_type)
        #x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var  


    def forward(self, x_T, y, method='ddim', skip=10,return_intermediate=False):
        """
        Algorithm 2.
            - method: sampling method, default='ddpm'
            - skip: decrease sampling steps from T/skip, default=1
        """
        x_t = x_T
        if return_intermediate:
            xt_list = []

        for time_step in reversed(range(0, self.T,skip)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t, y=y, method=method, skip=skip)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0

            if method == 'ddim':
                # ODE for DDIM
                x_t = mean
            else:
                # SDE for DDPM
                x_t = mean + torch.exp(0.5 * log_var) * noise
                # # delete this line
                # x_t_Guided=mean_Guided + torch.exp(0.5 * log_var_Guided) * noise
            if return_intermediate:
                xt_list.append(x_t.cpu())

            # update guidance in every step
            #x_t = mean + torch.exp(0.5 * log_var) * noise
        x_0 = x_t
        if return_intermediate:
            return torch.clip(x_0, -1, 1),xt_list
        return torch.clip(x_0, -1, 1)


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def uniform_sampling(n, N, k):
    return np.stack([np.random.randint(int(N/n)*i, int(N/n)*(i+1), k) for i in range(n)])


def dist(X, Y):
    sx = torch.sum(X**2, dim=1, keepdim=True)
    sy = torch.sum(Y**2, dim=1, keepdim=True)
    return torch.sqrt(-2 * torch.mm(X, Y.T) + sx + sy.T)


def topk(y, all_y, K):
    dist_y = dist(y, all_y)
    return torch.topk(-dist_y, K, dim=1)[1]


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self,
                 model, beta_1, beta_T, T, dataset,
                 num_class, cfg, weight,transfer_x0=True,
                 mixing=False,transfer_mode='full',transfer_only_uncond=False,
                 transfer_label=False,transfer_tr_tau=False,label_weight_tr = None,
                 count=False,cut_time=-1,transfer_only_cond=False,uncond_flag_from_out=False,
                 double_transfer=False):
        super().__init__()

        self.model = model
        self.T = T
        self.dataset = dataset
        self.num_class = num_class
        self.cfg = cfg
        self.transfer_mode = transfer_mode
        self.weight = weight
        self.transfer_x0 = transfer_x0
        self.transfer_label=transfer_label
        self.transfer_only_uncond = transfer_only_uncond
        self.transfer_tr_tau = transfer_tr_tau
        self.label_weight_tr = label_weight_tr
        self.mixing = mixing
        self.count = count
        self.cut_time = cut_time
        if count:
            self.total_count = np.zeros(T)
            self.transfer_count = np.zeros(T)
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)


        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.register_buffer(
            'sigma_tsq', 1./alphas_bar-1.)
        self.register_buffer('sigma_t',torch.sqrt(self.sigma_tsq))

    def forward(self, x_0, y_0, augm=None,uncond_flag_out=False):
        """
        Algorithm 1.
        """
        # original codes
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0) 
        ini_noise = noise

        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        uncond_flag = False
        y_l = y_0

        if self.cfg:
            if torch.rand(1)[0] < 1/10:
                y_l = None
                uncond_flag = True
            else:
                y_l = y_0

        h = self.model(x_t, t, y=y_l, augm=augm)
        if self.transfer_x0:
            cx_t = x_0 + extract(self.sigma_t, t, x_0.shape) * noise
            if self.transfer_tr_tau:
                noise = self.do_transfer_x0_with_y(x_t,cx_t,x_0,t,y_0,self.label_weight_tr)
            else:
                noise,_ = self.do_transfer_x0(x_t,cx_t,x_0,t,y_0,return_transfer_label=True)

        loss = F.mse_loss(h, noise, reduction='none')
        loss_reg = loss_com = torch.tensor(0).to(x_t.device)

        return loss, loss_reg + 1/4 * loss_com

    def do_transfer_x0(self,x_t,cx_t,x_0,t,y,return_transfer_label=False,mode=None,x_ref=None):
        '''
        new item for this function:
        restrict the transfer direction from long to tail or tail to long.
        '''
        if mode is not None:
            this_mode = mode
        else:
            this_mode = self.transfer_mode
        with torch.no_grad():
            bs,ch,h,w = x_0.shape
            ### here we should change the defination of the x_t

            x_t1 = cx_t.reshape(len(x_t),-1)
            x_01 = x_0.reshape(len(x_0),-1)
            '''
            here we should decay the initial signal by sqrt{alpha_t}
            '''
            com_dis = x_t1.unsqueeze(1) - x_01
            gt_distance = torch.sum((x_t1.unsqueeze(1) - x_01)**2,dim=[-1])
            normalize_distance = 2*extract(self.sigma_tsq, t, gt_distance.shape)

            #distance = - torch.max(gt_distance, dim=1, keepdim=True)[0] + gt_distance
            gt_distance = - gt_distance / normalize_distance
            distance = - torch.max(gt_distance, dim=1, keepdim=True)[0] + gt_distance
            distance = torch.exp(distance)
            # add y prior knowledge
            # self-normalize the per-sample weight of reference batch
            weights = distance / (torch.sum(distance, dim=1, keepdim=True))

            new_ind = torch.multinomial(weights,1)
            # here we wanted to record the transfer probability


            new_ind = new_ind.squeeze(); ini_ind = torch.arange(x_0.shape[0]).cuda()
            transfer_label = y[new_ind]
            old_prob = self.weight.squeeze().cuda().gather(0,y)
            new_prob = self.weight.squeeze().cuda().gather(0,transfer_label)
            #here add the restriction item, just make judgement!
            if this_mode == 't2h':
                # ipdb.set_trace()
                # here we implement the long to tail transfer
                # firstly we should obtain the y label to the corresponding images
                # initial label is the y 
                if self.cut_time < 0:
                    new_ind_f = torch.where(new_prob>=old_prob,new_ind,ini_ind)
                else:
                    new_ind_f1 = torch.where(new_prob>=old_prob ,new_ind,ini_ind)
                    new_ind_f = torch.where(t < self.cut_time,new_ind_f1,ini_ind)
            elif this_mode == 'h2t':
                if self.cut_time < 0:
                    new_ind_f = torch.where(new_prob<=old_prob,new_ind,ini_ind)
                else:
                    new_ind_f1 = torch.where(new_prob<=old_prob,new_ind,ini_ind)
                    new_ind_f = torch.where(t < self.cut_time,new_ind_f1,ini_ind)
            elif this_mode == 'full':
                new_ind_f = new_ind
            else:
                raise NotImplementedError
            x_n0 = x_0[new_ind_f]

            new_epsilon =  (x_t - extract(self.sqrt_alphas_bar, t, x_0.shape)*x_n0) / extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape)
                        # then record the things 
            if self.count:
                reference = list(range(len(new_ind_f)))
                for i in range(len(reference)):
                    if new_ind_f[i] == reference[i]:
                        self.total_count[t[i].item()] +=1
                    else:
                        self.total_count[t[i].item()] +=1
                        self.transfer_count[t[i].item()]+=1

            if return_transfer_label:
                return new_epsilon,transfer_label
            return new_epsilon

    def do_transfer_x0_with_y(self,x_t,cx_t,x_0,t,y,weight_label):
        '''
        new item for this function:
        restrict the transfer direction from long to tail or tail to long.
        '''
        with torch.no_grad():
            bs,ch,h,w = x_0.shape
            ### here we should change the defination of the x_t

            x_t1 = cx_t.reshape(len(x_t),-1)
            x_01 = x_0.reshape(len(x_0),-1)
            '''
            here we should decay the initial signal by sqrt{alpha_t}
            '''
            com_dis = x_t1.unsqueeze(1) - x_01
            gt_distance = torch.sum((x_t1.unsqueeze(1) - x_01)**2,dim=[-1])
            normalize_distance = 2*extract(self.sigma_tsq, t, gt_distance.shape)

            #distance = - torch.max(gt_distance, dim=1, keepdim=True)[0] + gt_distance
            gt_distance = - gt_distance / normalize_distance
            distance = - torch.max(gt_distance, dim=1, keepdim=True)[0] + gt_distance
            wl = weight_label.cuda()
            reweight = torch.gather(wl[y],1,y.unsqueeze(0).repeat(bs,1))
                         #distance = torch.exp(distance) * weight_label
            distance = reweight * torch.exp(distance)#distance
            # self-normalize the per-sample weight of reference batch
            weights = distance / (torch.sum(distance, dim=1, keepdim=True))
            new_ind = torch.multinomial(weights,1)
            new_ind = new_ind.squeeze(); ini_ind = torch.arange(x_0.shape[0]).cuda()

            new_ind_f = new_ind
            x_n0 = x_0[new_ind_f]
            new_epsilon =  (x_t - extract(self.sqrt_alphas_bar, t, x_0.shape)*x_n0) / extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape)

                        # then record the things 
            return new_epsilon





class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, num_class, img_size=32, var_type='fixedlarge'):
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.num_class = int(num_class)
        self.img_size = img_size
        self.var_type = var_type
        
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer(
            'alphas_bar', alphas_bar)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps): 
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    ### May change it to cg mode.
    

    def p_mean_variance(self, x_t, t, y=None, omega=0.0, method='free'):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped}[self.var_type]

        model_log_var = extract(model_log_var, t, x_t.shape)
        unc_eps = None
        augm = torch.zeros((x_t.shape[0], 9)).to(x_t.device)

        # Mean parameterization
        eps = self.model(x_t, t, y=y, augm=augm)
        if omega > 0 and (method == 'cfg'):
            unc_eps = self.model(x_t, t, y=None, augm=None)
            guide = eps - unc_eps
            eps = eps + omega * guide
        
        x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
        model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def forward(self, x_T, omega=0.0, method='cfg'):
        """
        Algorithm 2.
        """
        x_t = x_T.clone()
        y = None

        if method == 'uncond':
            y = None
        else:
            y = torch.randint(0, self.num_class, (len(x_t),)).to(x_t.device)

        with torch.no_grad():
            for time_step in tqdm(reversed(range(0, self.T)), total=self.T):
                t = x_T.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                mean, log_var = self.p_mean_variance(x_t=x_t, t=t, y=y,
                                                     omega=omega, method=method)

                if time_step > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0
                
                x_t = mean + torch.exp(0.5 * log_var) * noise

        return torch.clip(x_t, -1, 1), y



class GaussianDiffusionSamplerCond(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='eps', var_type='fixedlarge',w=2,cond=False):
        # assert mean_type in ['xprev' 'xstart', 'eps']
        # assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type
        self.cond = cond
        self.w=w
        print(f"current guidance rate is {w}")
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer(
            'alphas_bar', alphas_bar)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t,
                        method='ddpm',
                        skip=1,
                        eps=None):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape

        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)

        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape)

    def p_mean_variance(self, x_t, t, y,method, skip):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)


        eps = self.model(x_t, t, y)
        x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
        #x_0 = x_0.clamp(-1,1)
        model_mean, _ = self.q_mean_variance(x_0, x_t, t, method='ddpm', skip=10, eps=eps)
        #x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var,x_0,eps



    def condition_score(self, cond_fn, x_0, x, t, y,method='ddim',skip=10):
        """
        Borrow from guided diffusion "Diffusion Beat Gans in Image Synthesis"
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = extract(self.alphas_bar, t, x.shape)
        eps = self._predict_eps_from_xstart(x, t, x_0)
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, y)


        # out = p_mean_var.copy()
        # out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        # out["mean"], _, _ = self.q_posterior_mean_variance(
        #     x_start=out["pred_xstart"], x_t=x, t=t
        # )
        cond_x0 = self.predict_xstart_from_eps(x, t, eps)
        cond_mean, _ = self.q_mean_variance(cond_x0, x, t,
                        method='ddpm',
                        skip=skip,
                        eps=eps)

        return cond_x0,cond_mean


    def forward(self, x_T, y, method='ddim', skip=10,cond_fn=None):
        """
        Algorithm 2.
            - method: sampling method, default='ddpm'
            - skip: decrease sampling steps from T/skip, default=1
        """
        x_t = x_T

        for time_step in reversed(range(0, self.T,skip)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            t = t.to(x_t.device)
            mean, log_var, pred_x0, eps = self.p_mean_variance(x_t=x_t, t=t, y=y, method='ddpm', skip=skip)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0

            if method == 'ddim':
                #x_t = mean
                # ODE for DDIM
                pred_x0, cond_mean = self.condition_score(cond_fn, pred_x0, mean, t, y,method='ddpm',skip=skip)
                eps = self._predict_eps_from_xstart(x_t,t,pred_x0)
                # x_t, _ = self.q_mean_variance(pred_x0, x_t, t,method='ddim',skip=skip,eps=eps)
                
                assert (eps is not None)
                skip_time = torch.clamp(t - skip, 0, self.T)
                posterior_mean_coef1 = torch.sqrt(extract(self.alphas_bar, t, x_t.shape))
                posterior_mean_coef2 = torch.sqrt(1-extract(self.alphas_bar, t, x_t.shape))
                posterior_mean_coef3 = torch.sqrt(extract(self.alphas_bar, skip_time, x_t.shape))
                posterior_mean_coef4 = torch.sqrt(1-extract(self.alphas_bar, skip_time, x_t.shape))
                x_t = (
                    posterior_mean_coef3 / posterior_mean_coef1 * x_t +
                    (posterior_mean_coef4 - 
                    posterior_mean_coef3 * posterior_mean_coef2 / posterior_mean_coef1) * eps
                )


            else:
                # SDE for DDPM
                x_t = mean + torch.exp(0.5 * log_var) * noise
                # # delete this line
                # x_t_Guided=mean_Guided + torch.exp(0.5 * log_var_Guided) * noise

            # update guidance in every step
            #x_t = mean + torch.exp(0.5 * log_var) * noise

        x_0 = x_t

        return torch.clip(x_0, -1, 1),y


class GaussianDiffusionClassifier(nn.Module):
    def __init__(self,
                 model, beta_1, beta_T, T, dataset,
                 num_class,loss_type='softmax',sample_per_class=None):
        super().__init__()

        self.model = model
        self.T = T
        self.dataset = dataset
        self.num_class = num_class
        self.loss_type = loss_type
        if loss_type == 'softmax':
            self.loss = nn.CrossEntropyLoss(reduction='mean')
            print("using cross entropy!")
        elif loss_type == 'balancedsoftmax':
            self.loss = partial(self.balanced_softmax_loss,sample_per_class=sample_per_class,reduction='mean')
            print("using balanced ce!")
        elif loss_type == 'pll_cc':
            self.loss = self.cc_loss
            print("using partial label!")
        else:
            raise NotImplementedError
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, y_0, return_logit=False,fix_t=None):
        """
        Algorithm 1.
        """
        # original codes
        if fix_t is None:
            t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        else:
            t = x_0.new_ones([x_0.shape[0], ], dtype=torch.long) * fix_t
        noise = torch.randn_like(x_0) 

        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)

        y_pred = self.model(x_t, t)
        if return_logit:
            return y_pred

        if self.loss_type == 'pll_cc':
            loss = self.loss(y_pred,y_0,t)
        else:
            loss = self.loss(y_pred,y_0)
        return loss
    
    def balanced_softmax_loss(self, logits,labels, sample_per_class, reduction):
        """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
        Args:
          labels: A int tensor of size [batch].
          logits: A float tensor of size [batch, no_of_classes].
          sample_per_class: A int tensor of size [no of classes].
          reduction: string. One of "none", "mean", "sum"
        Returns:
          loss: A float tensor. Balanced Softmax Loss.
        """
        spc = sample_per_class.type_as(logits)
        spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
        logits = logits + spc.log()
        loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
        return loss

    def cc_loss(self,outputs, partialY,time):
        # partialY shape [N,class,Time]
        bs = partialY.shape[0]
        n_class = partialY.shape[1]
        sm_outputs = F.softmax(outputs, dim=1)
        time = time.unsqueeze(1).unsqueeze(1).expand(bs,n_class,1)
        this_partial_Y = torch.gather(partialY,index=time,dim=2)
        final_outputs = sm_outputs * this_partial_Y.squeeze()
        average_loss = -torch.log(final_outputs.sum(dim=1)).mean()
        return average_loss


class RSGClassifier(nn.Module):
    def __init__(self,
                 model):
        super().__init__()

        self.model = model 
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        from RSG import RSG
        self.RSG = RSG(feature_maps_shape=[48,4,4])
        ## followed by a non_linear classifier


    def forward(self, x_0, y_0,head_class_lists,epoch):
        """
        Algorithm 1.
        """
        # original codes
        bs,C,H,W = x_0.shape

        new_x_0 = F.avg_pool2d(x_0,kernel_size=(2,2))
        new_x_0 = new_x_0.view(bs,C*16,4,4)

        new_aug_x_0, loss_cesc, loss_mv_total, target = self.RSG(new_x_0, head_class_lists, y_0, epoch)
        new_aug_x_0 = new_aug_x_0.view(bs,C,16,-1)
        new_aug_x_0 = F.interpolate(new_aug_x_0,scale_factor=2)
        y_pred = self.model(new_aug_x_0)
        loss = self.loss(y_pred,target)
        return loss + 0.1 * loss_cesc.mean() + 0.01 * loss_cesc.mean()



class DoubleGaussianDiffusionSampler(nn.Module):
    def __init__(self, model_cond,model_uncond, beta_1, beta_T, T, img_size=32,
                 mean_type='epsilon', var_type='fixedlarge',w=2,cond=False,cut_time=800):
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model_cond = model_cond
        self.model_uncond = model_uncond
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type
        self.cond = cond
        self.cut_time = cut_time
        self.w=w
        print(f"current guidance rate is {w}")
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer(
            'alphas_bar', alphas_bar)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t,
                        method='ddpm',
                        skip=1,
                        eps=None):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        if method == 'ddim':
            assert (eps is not None)
            skip_time = torch.clamp(t - skip, 0, self.T)
            posterior_mean_coef1 = torch.sqrt(extract(self.alphas_bar, t, x_t.shape))
            posterior_mean_coef2 = torch.sqrt(1-extract(self.alphas_bar, t, x_t.shape))
            posterior_mean_coef3 = torch.sqrt(extract(self.alphas_bar, skip_time, x_t.shape))
            posterior_mean_coef4 = torch.sqrt(1-extract(self.alphas_bar, skip_time, x_t.shape))
            posterior_mean = (
                posterior_mean_coef3 / posterior_mean_coef1 * x_t +
                (posterior_mean_coef4 - 
                posterior_mean_coef3 * posterior_mean_coef2 / posterior_mean_coef1) * eps
            )
        else:
            posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)

        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t, y,method, skip):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t, y)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t, t ,y)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            if self.cond:
                if t[0] > self.cut_time:
                    eps=self.model_uncond(x_t, t ,None)
                    x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
                    model_mean, _ = self.q_mean_variance(x_0, x_t, t, method, skip, eps)
                else:
                    eps = self.model_cond(x_t, t ,y)
                    eps_g=self.model_cond(x_t, t ,None)
                    eps=eps+(self.w)*(eps-eps_g)
                    x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
                    model_mean, _ = self.q_mean_variance(x_0, x_t, t, method, skip, eps)
            else:
                #ipdb.set_trace()
                eps = self.model(x_t, t)
                x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
                model_mean, _ = self.q_mean_variance(x_0, x_t, t, method, skip, eps)
                #print("un conditional!")
        else:
            raise NotImplementedError(self.mean_type)
        #x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var  


    def forward(self, x_T, y, method='ddim', skip=10):
        """
        Algorithm 2.
            - method: sampling method, default='ddpm'
            - skip: decrease sampling steps from T/skip, default=1
        """
        x_t = x_T

        for time_step in reversed(range(0, self.T,skip)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t, y=y, method=method, skip=skip)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0

            if method == 'ddim':
                # ODE for DDIM
                x_t = mean
            else:
                # SDE for DDPM
                x_t = mean + torch.exp(0.5 * log_var) * noise
                # # delete this line
                # x_t_Guided=mean_Guided + torch.exp(0.5 * log_var_Guided) * noise

            # update guidance in every step
            #x_t = mean + torch.exp(0.5 * log_var) * noise

        x_0 = x_t

        return torch.clip(x_0, -1, 1)




