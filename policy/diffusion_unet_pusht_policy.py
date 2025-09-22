from array import array
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Callable
import math
from types import SimpleNamespace

from model.pusht_encoder import PushTObsEncoder, create_pusht_encoder
from model.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import numpy as np

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

class DiffusionUnetPushTPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Create UNet for noise prediction
        self.unet = ConditionalUnet1D(
            input_dim=cfg.action_dim,  # 2 for PushT (x, y)
            global_cond_dim=cfg.global_cond_dim,  # 条件维度
            diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
            down_dims=cfg.down_dims,
            kernel_size=cfg.kernel_size,
            n_groups=cfg.n_groups,
        )
        
        # Create observation encoder
        self.encoder = create_pusht_encoder(
            obs_horizon=cfg.obs_horizon,
            #output_dim=cfg.global_cond_dim,  # 确保输出维度匹配
            #use_group_norm=True,
            #imagenet_norm=getattr(cfg, 'imagenet_norm', False)
        )
        
        # Create noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.num_diffusion_steps,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            beta_schedule=cfg.beta_schedule,
            clip_sample=False,
            prediction_type="epsilon",
        )
        
        
        
        # Configuration
        self.obs_as_global_cond = True  # PushT使用全局条件
        self.n_obs_steps = cfg.obs_horizon
    
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            **kwargs):
        model = self.unet
        scheduler = self.noise_scheduler
        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
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
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    def set_normalizer(self, action_stats, pos_stats):
        # Store normalization stats
        #self.register_buffer('action_stats_min', action_stats['min'])
        #self.register_buffer('action_stats_max', action_stats['max'])
        #self.register_buffer('pos_stats_min', pos_stats['min'])
        #self.register_buffer('pos_stats_max', pos_stats['max'])
        self.action_stats_min = action_stats['min']
        self.action_stats_max = action_stats['max']
        self.pos_stats_min = pos_stats['min']
        self.pos_stats_max = pos_stats['max']

    def normalize_data(self, data, stats):
        # nomalize to [0,1]
        ndata = (data - stats['min']) / (stats['max'] - stats['min'])
        # normalize to [-1, 1]
        ndata = ndata * 2 - 1
        return ndata
        
    def unnormalize_data(self, ndata, stats):
        ndata = (ndata + 1) / 2
        data = ndata * (stats['max'] - stats['min']) + stats['min']
        return data

    def forward(self, trajectory: torch.Tensor, timesteps: torch.Tensor, global_cond: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Forward pass for the UNet model
        
        Args:
            trajectory: (B, pred_horizon, action_dim) noisy action trajectory
            timesteps: (B,) diffusion timesteps
            global_cond: (B, global_cond_dim) global conditioning features
            
        Returns:
            noise_pred: (B, pred_horizon, action_dim) predicted noise
        """
        # Forward through UNet
        noise_pred = self.unet(trajectory, timesteps, global_cond)
        return noise_pred

    def compute_loss(self, batch):
        """
        Compute training loss following the reference implementation pattern
        
        Args:
            batch: Dictionary containing 'image', 'agent_pos', 'action'
            
        Returns:
            loss: MSE loss between predicted and actual noise
        """
        device = next(self.parameters()).device
        
        # Extract and move data to device
        obs_dict = {}
        for key, value in batch.items():
            if key not in ['action']:
                obs_dict[key] = value.to(device, non_blocking=True)
        
        # Normalize actions
        raw_actions = batch['action'].to(device, non_blocking=True)  # (B, pred_horizon, action_dim)
        # nactions = self.normalize_action(raw_actions)
        nactions = raw_actions  # already normalized in dataset
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]  # pred_horizon
        
        # Handle observation encoding as global conditioning
        local_cond = None
        global_cond = None
        trajectory = nactions  # The action trajectory to denoise
        
        if self.obs_as_global_cond:
            global_cond = self.encoder(obs_dict)  # (B, global_cond_dim)
        else:
            raise NotImplementedError("Local conditioning not implemented for PushT")

        # Sample noise that we'll add to the trajectory
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        
        # Sample random timesteps for each sample in the batch
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=trajectory.device
        ).long()
        
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        
        # Predict the noise residual
        noise_pred = self(noisy_trajectory, timesteps, global_cond=global_cond)
        
        # Compute loss based on prediction type
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            print('Caution: using sample prediction type!')
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        
        # Compute MSE loss
        loss = F.mse_loss(noise_pred, target, reduction='mean')
        
        return loss

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generate action sequence given observations using DDPM sampling
        
        Args:
            obs_dict: dictionary containing 'image' and 'agent_pos'
            
        Returns:
            action: (1, pred_horizon, action_dim) predicted actions
        """
        device = next(self.parameters()).device
        batch_size = 1
        
        # Encode observations as global conditioning
        global_cond = self.encoder(obs_dict)  # (1, global_cond_dim)
        
        # Initialize trajectory with random noise
        action_shape = (batch_size, self.cfg.pred_horizon, self.cfg.action_dim)
        trajectory = torch.randn(action_shape, device=device)
        
        # Set up the noise scheduler for inference
        self.noise_scheduler.set_timesteps(self.cfg.num_diffusion_steps)
        
        # Denoising loop
        for t in self.noise_scheduler.timesteps:
            # Prepare timestep tensor
            timestep = t.expand(batch_size).to(device)
            
            # Predict noise
            noise_pred = self(trajectory, timestep, global_cond=global_cond)
            
            # Denoising step
            trajectory = self.noise_scheduler.step(noise_pred, t, trajectory).prev_sample
        
        # Unnormalize actions to original scale

        action = self.unnormalize_data(trajectory.detach().cpu().numpy(), stats={'min': self.action_stats_min, 'max': self.action_stats_max})
        
        return action


def create_pusht_policy_config():
    """Create default config for PushT policy - 使用SimpleNamespace"""
    config = SimpleNamespace(
        # Action space
        action_dim=2,
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        
        # UNet architecture - 使用ConditionalUnet1D的实际参数
        global_cond_dim=1028,  # 条件特征维度
        diffusion_step_embed_dim=256,  # 时间步编码维度
        down_dims=[256, 512, 1024],  # UNet各层的通道数
        kernel_size=5,  # 卷积核大小
        n_groups=8,  # GroupNorm的组数
        
        # Diffusion scheduler
        num_diffusion_steps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        
        # Data preprocessing
        imagenet_norm=False,
    )
        
    return config


if __name__ == "__main__":
    # Test policy creation
    cfg = create_pusht_policy_config()
    policy = DiffusionUnetPushTPolicy(cfg)
    
    # Test with dummy data
    batch_size = 4
    batch = {
        'image': torch.randn(batch_size, cfg.obs_horizon, 3, 96, 96),
        'agent_pos': torch.randn(batch_size, cfg.obs_horizon, 2),
        'action': torch.randn(batch_size, cfg.pred_horizon, cfg.action_dim)
    }
    
    # Set dummy normalizer
    policy.action_stats_min = torch.zeros(cfg.action_dim)
    policy.action_stats_max = torch.ones(cfg.action_dim)
    
    # Test compute_loss
    loss = policy.compute_loss(batch)
    print(f"Training loss: {loss.item():.4f}")
    
    # Test action prediction
    obs_dict = {
        'image': batch['image'][:1],
        'agent_pos': batch['agent_pos'][:1]
    }
    predicted_action = policy.predict_action(obs_dict)
    print(f"Predicted action shape: {predicted_action.shape}")
    
    print("PushT policy test successful!")