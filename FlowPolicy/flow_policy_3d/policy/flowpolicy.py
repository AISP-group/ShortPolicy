import sys

from flow_policy_3d.model.flow.positional_embedding import SinusoidalPosEmb

sys.path.append('FlowPolicy/flow_policy_3d')
from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
import copy
import time
import numpy as np
from flow_policy_3d.sde_lib import ConsistencyFM
from flow_policy_3d.model.common.normalizer import LinearNormalizer
from flow_policy_3d.policy.base_policy import BasePolicy
from flow_policy_3d.model.flow.conditional_unet1d import ConditionalUnet1D
from flow_policy_3d.model.flow.mask_generator import LowdimMaskGenerator
from flow_policy_3d.common.pytorch_util import dict_apply
from flow_policy_3d.common.model_util import print_params
from flow_policy_3d.model.vision.pointnet_extractor import FlowPolicyEncoder
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm


class FlowPolicy(BasePolicy):
    def __init__(self, 
            shape_meta: dict, 
            horizon, 
            n_action_steps, 
            n_obs_steps,
            noise_scheduler: DDIMScheduler,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="mlp",
            pointcloud_encoder_cfg=None,
            Conditional_ConsistencyFM=None,           
            eta=0.01,
            sampling_method="euler",
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: 
            # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
        
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])
        
        # point cloud encoder
        obs_encoder = FlowPolicyEncoder(observation_space=obs_dict,
                                                   img_crop_shape=crop_shape,
                                                out_channel=encoder_output_dim,
                                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                use_pc_color=use_pc_color,
                                                pointnet_type=pointnet_type,
                                                )

        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        #obs_as_global_cond=true
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[FlowUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[FlowUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")


        model = ConditionalUnet1D(
            input_dim=input_dim, #128
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        
        
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        
        if Conditional_ConsistencyFM is None:
                    Conditional_ConsistencyFM = {
                        'eps': 1e-2,
                        'num_segments': 2,
                        'boundary': 1,
                        'delta': 1e-2,
                        'alpha': 1e-5,#
                        'num_inference_step': 1
                    }
        self.eta = eta
        self.eps = Conditional_ConsistencyFM['eps']
        self.num_segments = Conditional_ConsistencyFM['num_segments']
        self.boundary = Conditional_ConsistencyFM['boundary']
        self.delta = Conditional_ConsistencyFM['delta']
        self.alpha = Conditional_ConsistencyFM['alpha']
        self.num_inference_step = Conditional_ConsistencyFM['num_inference_step']

        print_params(self)


    def conditional_sample(
            self,
            condition_data,
            local_cond=None,
            global_cond=None,
            **kwargs,):

        noise = 1 * torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=None,)

        x = noise.detach().clone()

        t = torch.linspace(0, 1, steps=1).to(condition_data.device)

        x = odeint(lambda t, x: self.model(x,t * 100,local_cond=local_cond,global_cond=global_cond,),
            x,
            t,
            method=self.sampling_method,
            atol=1e-3,
            rtol=1e-3,)[-1]
        return x
    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            #print(f'1 : {nobs_features.shape}')#2,128
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True
        
        # run sampling
        noise = torch.randn(
            size=cond_data.shape,
            dtype=cond_data.dtype,
            device=cond_data.device,
            generator=None)
        z = noise.detach().clone() # a0

        sde = ConsistencyFM('gaussian',
                            noise_scale=1.0,
                            use_ode_sampler='rk45', # unused
                            sigma_var=0.0,
                            ode_tol=1e-5,
                            sample_N= self.num_inference_step)

        # Uniform
        dt = 1./self.num_inference_step
        eps = self.eps

        for i in range(sde.sample_N):
            num_t = i /sde.sample_N * (1 - eps) + eps
            t = torch.ones(z.shape[0], device=noise.device) * num_t
            pred = self.model(z, t*99, local_cond=local_cond, global_cond=global_cond) ### Copy from models/utils.py
            # convert to diffusion models if sampling.sigma_variance > 0.0 while perserving the marginal probability

            sigma_t = sde.sigma_t(num_t)
            pred_sigma = pred + (sigma_t**2)/(2*(sde.noise_scale**2)*((1.-num_t)**2)) * (0.5 * num_t * (1.-num_t) * pred - 0.5 * (2.-num_t)*z.detach().clone())
            z = z.detach().clone() + pred_sigma * dt + sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma).to(device)
        z[cond_mask] = cond_data[cond_mask] # a1

        # unnormalize prediction
        naction_pred = z[...,:Da]



        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        result = {
            'action': action,
            'action_pred': action_pred,
        }
        return result
    
    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def compute_loss(self, batch,rectified=False):
        eps = self.eps
        num_segments = self.num_segments
        boundary = self.boundary
        delta  = self.delta
        alpha =  self.alpha
        reduce_op = torch.mean
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        target = nactions

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()
        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)



        # ##----------------------------shortcut----------------------------------------
        x0 = torch.randn(trajectory.shape, device=trajectory.device)
        t = torch.rand(target.shape[0], device=target.device) * (1 - eps) + eps

        dt_base = 7-torch.tensor(np.arange(0, 8), device=target.device)
        dt_base = torch.cat([dt_base, torch.zeros(batch_size - dt_base.shape[0], device=target.device)])
        dt = 1 / (2 ** (dt_base))
        dt = dt / 2

        t2 = torch.clamp(t + dt, max=1.0)

        t_expand = t.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        t2_expand = t2.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        x_t = t_expand * target + (1. - t_expand) * x0
        x_t[condition_mask] = cond_data[condition_mask]

        v_t = self.model(x_t, t*99, local_cond=local_cond, global_cond=global_cond)
        x_t2 = x_t + (t2_expand - t_expand) * v_t #dt_expand
        x_t2[condition_mask] = cond_data[condition_mask]
        v_t2 = self.model(x_t2, t2*99, local_cond=local_cond, global_cond=global_cond)

        v_t[condition_mask] = cond_data[condition_mask]
        v_t2[condition_mask] = cond_data[condition_mask]
        v_t2 = torch.nan_to_num(v_t2)

        target_fm = target-x0
        target_sc = (v_t + v_t2) / 2
        loss_fm = torch.square(v_t - target_fm)
        loss_sc = torch.square(v_t - target_sc)
        loss = torch.mean(loss_fm + loss_sc )


        loss_dict = { 'bc_loss': loss.item(),}

        return loss, loss_dict