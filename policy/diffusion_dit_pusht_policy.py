import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Callable
from collections import defaultdict
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from model.pusht_encoder import create_pusht_encoder
from model.transformer_for_diffusion import TransformerForDiffusion, LowdimMaskGenerator
import os


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

class DiffusionDiTPushTPolicy(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        
        # config
        self.cfg = config
        policy_cfg = config['policy']
        noise_scheduler_cfg = config['noise_scheduler']

        obs_as_global_cond = policy_cfg.get('obs_as_global_cond', False)
        self.obs_as_global_cond = obs_as_global_cond
        shape_meta = config['shape_meta']
        action_shape = shape_meta['action']['shape']
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        self.n_obs_steps = policy_cfg.get('n_obs_steps', 2)
        #   obs encoder
        obs_encoder = create_pusht_encoder(
            obs_horizon=self.cfg.get('obs_horizon', 2),
        )
        self.obs_encoder = obs_encoder

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * policy_cfg.get('n_obs_steps', 2)

        model = TransformerForDiffusion(
            input_dim=policy_cfg.get('input_dim', 2),
            output_dim=policy_cfg.get('output_dim', 2),
            horizon=policy_cfg.get('horizon', 16),
            n_obs_steps=policy_cfg.get('n_obs_steps', 2),
            cond_dim=policy_cfg.get('cond_dim', 1028),
            n_layer=policy_cfg.get('n_layer', 8),
            n_head=policy_cfg.get('n_head', 8),
            n_emb=policy_cfg.get('n_emb', 512),
            p_drop_emb=policy_cfg.get('p_drop_emb', 0.1),
            p_drop_attn=policy_cfg.get('p_drop_attn', 0.1),
            causal_attn=policy_cfg.get('causal_attn', True),
            time_as_cond=policy_cfg.get('time_as_cond', True),
            obs_as_cond=obs_as_global_cond,
            n_cond_layers=policy_cfg.get('n_cond_layers', 4)
        )

        self.model = model
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_cfg.get('num_diffusion_steps', 100),
            beta_start=noise_scheduler_cfg.get('beta_start', 0.0001),
            beta_end=noise_scheduler_cfg.get('beta_end', 0.02),
            beta_schedule=noise_scheduler_cfg.get('beta_schedule', "squaredcos_cap_v2"),
            clip_sample=noise_scheduler_cfg.get('clip_sample', False),
            prediction_type=noise_scheduler_cfg.get('prediction_type', "epsilon"),
        )

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_global_cond) else obs_feature_dim,
            max_n_obs_steps=policy_cfg.get('n_obs_steps', 2),
            fix_obs_steps=True,
            action_visible=False
        )

        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.horizon = policy_cfg.get('horizon', 16)
        self.n_action_steps = policy_cfg.get('action_horizon', 8)
        self.num_inference_steps = policy_cfg.get('num_inference_steps', 100)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        batch: {
            'image': (B, obs_horizon, C, H, W)
            'agent_pos': (B, obs_horizon, 2)
            'action': (B, action_horizon, action_dim)
        }
        """
        device = next(self.parameters()).device
        nobs = {'image': batch['image'], 'agent_pos': batch['agent_pos']}
        nobs = dict_apply(nobs, lambda x: x.to(device))
        # nobs = batch['obs']
        nactions = batch['action'].to(device)


        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        To = self.n_obs_steps

        cond = None
        trajectory = nactions

        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            cond = nobs_features.reshape(batch_size, To, -1)
            #if self.pred_action_steps_only:
            #    start = To - 1
            #    end = start + self.n_action_steps
            #    trajectory = nactions[:,start:end]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()

        # generate impainting mask
        # condition_mask = self.mask_generator(trajectory.shape)
        condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.get('num_train_timesteps', 100), 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
    
    def set_normalizer(self, action_stats, pos_stats):
        self.action_stats_min = action_stats['min']
        self.action_stats_max = action_stats['max']
        self.pos_stats_min = pos_stats['min']
        self.pos_stats_max = pos_stats['max']
        self.pos_stats = pos_stats
        self.action_stats = action_stats

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

    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)

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
        # nobs = self.normalizer.normalize(obs_dict)
        device = next(self.parameters()).device
        #agent_poses = obs_dict['agent_pos']
        #nagent_poses = self.normalize_data(agent_poses, self.pos_stats)
        #obs_dict['agent_pos'] = nagent_poses
        nobs = dict_apply(obs_dict, lambda x: x.to(device))

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            cond = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da)
           
            cond_data = torch.zeros(size=shape, device=device, dtype=torch.float32)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=torch.float32)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            )
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.unnormalize_data(naction_pred.detach().cpu().numpy(), self.action_stats)
        

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result



def use_dummy_test():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_path, "config/pusht_dit.yaml")
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)  
    model = DiffusionDiTPushTPolicy(config)
    # print(model)

    # dummy input
    obs_dict = {
        'image': torch.randn(4, 2, 3, 64, 64),
        'agent_pos': torch.randn(4, 2, 2)
    }
    batch_input = {
        'obs': obs_dict,
        'action': torch.randn(4, 16, 2)
    }
    out = model.compute_loss(batch_input)
    #print(out)

if __name__ == "__main__":
    use_dummy_test()